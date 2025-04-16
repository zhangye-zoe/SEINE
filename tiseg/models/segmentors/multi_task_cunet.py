import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import measure
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects

from tiseg.utils import resize
from ..backbones import TorchVGG16BN
from ..builder import SEGMENTORS
from ..heads import MultiTaskUNetHead
from ..losses import MultiClassDiceLoss, BatchMultiClassDiceLoss, mdice, tdice
from ..utils import align_foreground
from .base import BaseSegmentor


@SEGMENTORS.register_module()
class MultiTaskCUNet(BaseSegmentor):
    """Base class for segmentors."""

    def __init__(self, num_classes, train_cfg, test_cfg):
        super(MultiTaskCUNet, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_classes = num_classes

        self.backbone = TorchVGG16BN(in_channels=3, pretrained=True, out_indices=[0, 1, 2, 3, 4, 5])
        self.head = MultiTaskUNetHead(
            num_classes=[3, self.num_classes],
            mt_dims=64,
            bottom_in_dim=512,
            skip_in_dims=(64, 128, 256, 512, 512),
            stage_dims=[16, 32, 64, 128, 256],
            act_cfg=dict(type='ReLU'),
            norm_cfg=dict(type='BN'))

    def calculate(self, img):
        img_feats = self.backbone(img)
        bottom_feat = img_feats[-1]
        skip_feats = img_feats[:-1]
        tc_logit, sem_logit = self.head(bottom_feat, skip_feats)

        # tc_logit: three-class semantic logit (background, nuclei, boundary)
        # sem_logit: semantic logit (background, nuclei class 1, nuclei class 2, ...)
        return tc_logit, sem_logit

    def forward(self, data, label=None, metas=None, **kwargs):
        """detectron2 style forward functions. Segmentor can be see as meta_arch of detectron2.
        """
        if self.training:
            tc_logit, sem_logit = self.calculate(data['img'])
            sem_gt = label['sem_gt']
            sem_gt_wb = label['sem_gt_w_bound']
            tc_gt = sem_gt_wb.clone()
            tc_gt[(tc_gt != 0) * (tc_gt != self.num_classes)] = 1
            tc_gt[tc_gt > 1] = 2
            loss = dict()
            sem_gt = sem_gt.squeeze(1)
            tc_gt = tc_gt.squeeze(1)
            sem_loss = self._sem_loss(sem_logit, sem_gt)
            loss.update(sem_loss)
            tc_loss = self._tc_loss(tc_logit, tc_gt)
            loss.update(tc_loss)
            # calculate training metric
            training_metric_dict = self._training_metric(sem_logit, sem_gt)
            loss.update(training_metric_dict)
            return loss
        else:
            assert metas is not None
            # NOTE: only support batch size = 1 now.
            tc_logit, sem_logit = self.inference(data['img'], metas[0], True)
            tc_pred = tc_logit.argmax(dim=1)
            sem_pred = sem_logit.argmax(dim=1)
            # Extract inside class
            tc_pred = tc_pred.cpu().numpy()[0]
            sem_pred = sem_pred.cpu().numpy()[0]
            sem_pred, inst_pred = self.postprocess(tc_pred, sem_pred)
            # unravel batch dim
            ret_list = []
            ret_list.append({'sem_pred': sem_pred, 'inst_pred': inst_pred})
            return ret_list

    def postprocess(self, tc_pred, sem_pred):
        """model free post-process for both instance-level & semantic-level."""
        sem_id_list = list(np.unique(sem_pred))
        sem_canvas = np.zeros_like(sem_pred).astype(np.uint8)
        for sem_id in sem_id_list:
            # 0 is background semantic class.
            if sem_id == 0:
                continue
            sem_id_mask = sem_pred == sem_id
            # fill instance holes
            sem_id_mask = remove_small_objects(sem_id_mask, 5)
            sem_id_mask = binary_fill_holes(sem_id_mask)
            sem_canvas[sem_id_mask > 0] = sem_id
        sem_pred = sem_canvas

        # instance process & dilation
        bin_pred = tc_pred.copy()
        bin_pred[bin_pred == 2] = 0

        inst_pred = measure.label(bin_pred, connectivity=1)
        # if re_edge=True, dilation pixel length should be 2
        # inst_pred = morphology.dilation(inst_pred, selem=morphology.disk(2))
        inst_pred = align_foreground(inst_pred, sem_canvas > 0, 20)

        return sem_pred, inst_pred

    def inference(self, img, meta, rescale):
        """Inference with split/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            meta (dict): Image info dict.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        assert self.test_cfg.mode in ['split', 'whole']

        self.rotate_degrees = self.test_cfg.get('rotate_degrees', [0])
        self.flip_directions = self.test_cfg.get('flip_directions', ['none'])
        tc_logit_list = []
        sem_logit_list = []
        img_ = img

        for rotate_degree in self.rotate_degrees:
            for flip_direction in self.flip_directions:
                img = self.tta_transform(img_, rotate_degree, flip_direction)

                # inference patch or whole img
                if self.test_cfg.mode == 'split':
                    tc_logit, sem_logit = self.split_inference(img, meta, rescale)
                else:
                    tc_logit, sem_logit = self.whole_inference(img, meta, rescale)

                tc_logit = self.reverse_tta_transform(tc_logit, rotate_degree, flip_direction)
                sem_logit = self.reverse_tta_transform(sem_logit, rotate_degree, flip_direction)

                tc_logit = F.softmax(tc_logit, dim=1)
                sem_logit = F.softmax(sem_logit, dim=1)

                tc_logit_list.append(tc_logit)
                sem_logit_list.append(sem_logit)

        tc_logit = sum(tc_logit_list) / len(tc_logit_list)
        sem_logit = sum(sem_logit_list) / len(sem_logit_list)

        if rescale:
            tc_logit = resize(tc_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            sem_logit = resize(sem_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)

        return tc_logit, sem_logit

    def split_inference(self, img, meta, rescale):
        """using half-and-half strategy to slide inference."""
        window_size = self.test_cfg.crop_size[0]
        overlap_size = self.test_cfg.overlap_size[0]

        B, C, H, W = img.shape

        # zero pad for border patches
        pad_h = 0
        if H - window_size > 0:
            pad_h = (window_size - overlap_size) - (H - window_size) % (window_size - overlap_size)
        else:
            pad_h = window_size - H

        if W - window_size > 0:
            pad_w = (window_size - overlap_size) - (W - window_size) % (window_size - overlap_size)
        else:
            pad_w = window_size - W

        H1, W1 = pad_h + H, pad_w + W
        img_canvas = torch.zeros((B, C, H1, W1), dtype=img.dtype, device=img.device)
        img_canvas[:, :, pad_h // 2:pad_h // 2 + H, pad_w // 2:pad_w // 2 + W] = img

        tc_logit = torch.zeros((B, 3, H1, W1), dtype=img.dtype, device=img.device)
        sem_logit = torch.zeros((B, self.num_classes, H1, W1), dtype=img.dtype, device=img.device)
        for i in range(0, H1 - overlap_size, window_size - overlap_size):
            r_end = i + window_size if i + window_size < H1 else H1
            ind1_s = i + overlap_size // 2 if i > 0 else 0
            ind1_e = (i + window_size - overlap_size // 2 if i + window_size < H1 else H1)
            for j in range(0, W1 - overlap_size, window_size - overlap_size):
                c_end = j + window_size if j + window_size < W1 else W1

                img_patch = img_canvas[:, :, i:r_end, j:c_end]
                tc_patch, sem_patch = self.calculate(img_patch)

                ind2_s = j + overlap_size // 2 if j > 0 else 0
                ind2_e = (j + window_size - overlap_size // 2 if j + window_size < W1 else W1)
                tc_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = tc_patch[:, :, ind1_s - i:ind1_e - i,
                                                                        ind2_s - j:ind2_e - j]
                sem_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = sem_patch[:, :, ind1_s - i:ind1_e - i,
                                                                          ind2_s - j:ind2_e - j]

        tc_logit = tc_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
        sem_logit = sem_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]

        return tc_logit, sem_logit

    def whole_inference(self, img, meta, rescale):
        """Inference with full image."""

        tc_logit, sem_logit = self.calculate(img)

        return tc_logit, sem_logit

    def _sem_loss(self, sem_logit, sem_gt):
        """calculate semantic mask branch loss."""
        sem_loss = {}

        # loss weight
        alpha = 5
        beta = 0.5

        sem_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        sem_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=self.num_classes)
        sem_ce_loss = torch.mean(sem_ce_loss_calculator(sem_logit, sem_gt))
        sem_dice_loss = sem_dice_loss_calculator(sem_logit, sem_gt)
        sem_loss['sem_ce_loss'] = alpha * sem_ce_loss
        sem_loss['sem_dice_loss'] = beta * sem_dice_loss

        return sem_loss

    def _tc_loss(self, tc_logit, tc_gt):
        """calculate three-class mask branch loss."""
        tc_loss = {}

        tc_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        tc_dice_loss_calculator = MultiClassDiceLoss(num_classes=3)
        tc_ce_loss = torch.mean(tc_ce_loss_calculator(tc_logit, tc_gt))
        tc_dice_loss = tc_dice_loss_calculator(tc_logit, tc_gt)
        # loss weight
        alpha = 5
        beta = 0.5
        tc_loss['three_class_ce_loss'] = alpha * tc_ce_loss
        tc_loss['three_class_dice_loss'] = beta * tc_dice_loss

        return tc_loss

    def _training_metric(self, sem_logit, sem_gt):
        """metric calculation when training."""
        wrap_dict = {}

        # loss
        clean_sem_logit = sem_logit.clone().detach()
        clean_sem_gt = sem_gt.clone().detach()

        wrap_dict['sem_tdice'] = tdice(clean_sem_logit, clean_sem_gt, self.num_classes)
        wrap_dict['sem_mdice'] = mdice(clean_sem_logit, clean_sem_gt, self.num_classes)

        # NOTE: training aji calculation metric calculate (This will be deprecated.)
        # (the edge id is set `self.num_classes - 1` in default)
        # sem_pred = torch.argmax(sem_logit, dim=1).cpu().numpy().astype(np.uint8)
        # sem_pred[sem_pred == (self.num_classes - 1)] = 0
        # sem_target = sem_gt.cpu().numpy().astype(np.uint8)
        # sem_target[sem_target == (self.num_classes - 1)] = 0

        # N = sem_pred.shape[0]
        # wrap_dict['aji'] = 0.
        # for i in range(N):
        #     aji_single_image = aggregated_jaccard_index(sem_pred[i], sem_target[i])
        #     wrap_dict['aji'] += 100.0 * torch.tensor(aji_single_image)
        # # distributed environment requires cuda tensor
        # wrap_dict['aji'] /= N
        # wrap_dict['aji'] = wrap_dict['aji'].cuda()

        return wrap_dict
