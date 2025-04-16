import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import morphology, measure
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes

from tiseg.utils import resize
from ..backbones import TorchVGG16BN
from ..heads.cd_head import CDHead
from ..builder import SEGMENTORS
from ..losses import BatchMultiClassDiceLoss, mdice, tdice
from ..utils import generate_direction_differential_map
from .base import BaseSegmentor


@SEGMENTORS.register_module()
class CDNet(BaseSegmentor):
    """Base class for segmentors."""

    def __init__(self, num_classes, train_cfg, test_cfg):
        super(CDNet, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_classes = num_classes
        self.num_angles = 8

        # argument
        self.if_mudslide = self.test_cfg.get('if_mudslide', False)

        self.backbone = TorchVGG16BN(in_channels=3, pretrained=True, out_indices=[0, 1, 2, 3, 4, 5])
        self.head = CDHead(
            num_classes=self.num_classes + 1,
            num_angles=self.num_angles,
            dgm_dims=64,
            bottom_in_dim=512,
            skip_in_dims=(64, 128, 256, 512, 512),
            stage_dims=[16, 32, 64, 128, 256],
            act_cfg=dict(type='ReLU'),
            norm_cfg=dict(type='BN'))

    def calculate(self, img, rescale=False):
        img_feats = self.backbone(img)
        bottom_feat = img_feats[-1]
        skip_feats = img_feats[:-1]
        sem_logit, dir_logit, point_logit = self.head(bottom_feat, skip_feats)

        return sem_logit, dir_logit, point_logit

    def forward(self, data, label=None, metas=None, **kwargs):
        """detectron2 style forward functions. Segmentor can be see as meta_arch of detectron2.
        """
        if self.training:
            sem_logit, dir_logit, point_logit = self.calculate(data['img'])

            assert label is not None
            sem_gt_wb = label['sem_gt_w_bound']
            point_gt = label['point_gt']
            dir_gt = label['dir_gt']
            weight_map = label['loss_weight_map'] if self.train_cfg.get('if_weighted_loss', False) else None

            loss = dict()

            sem_gt_wb = sem_gt_wb.squeeze(1)
            dir_gt = dir_gt.squeeze(1)

            # TODO: Conside to remove some edge loss value.
            # mask branch loss calculation
            sem_loss = self._sem_loss(sem_logit, sem_gt_wb, weight_map)
            loss.update(sem_loss)
            # direction branch loss calculation
            dir_loss = self._dir_loss(dir_logit, dir_gt, weight_map)
            loss.update(dir_loss)
            # point branch loss calculation
            point_loss = self._point_loss(point_logit, point_gt)
            loss.update(point_loss)

            # calculate training metric
            training_metric_dict = self._training_metric(sem_logit, dir_logit, point_logit, sem_gt_wb, dir_gt, point_gt)
            loss.update(training_metric_dict)
            return loss
        else:
            assert self.test_cfg is not None
            # NOTE: only support batch size = 1 now.
            sem_logit, dir_pred = self.inference(data['img'], metas[0], True)
            sem_pred = sem_logit.argmax(dim=1)
            sem_pred = sem_pred.to('cpu').numpy()[0]
            dir_pred = dir_pred.to('cpu').numpy()[0]
            sem_pred, inst_pred = self.postprocess(sem_pred)
            # # unravel batch dim
            ret_list = []
            ret_list.append({'sem_pred': sem_pred, 'inst_pred': inst_pred})
            return ret_list

    def postprocess(self, pred):
        """model free post-process for both instance-level & semantic-level."""
        pred[pred == self.num_classes] = 0
        sem_id_list = list(np.unique(pred))
        inst_pred = np.zeros_like(pred).astype(np.int32)
        sem_pred = np.zeros_like(pred).astype(np.uint8)
        cur = 0
        for sem_id in sem_id_list:
            # 0 is background semantic class.
            if sem_id == 0:
                continue
            sem_id_mask = pred == sem_id
            # fill instance holes
            sem_id_mask = binary_fill_holes(sem_id_mask)
            sem_id_mask = remove_small_objects(sem_id_mask, 5)
            inst_sem_mask = measure.label(sem_id_mask)
            inst_sem_mask = morphology.dilation(inst_sem_mask, selem=morphology.disk(self.test_cfg.get('radius', 3)))
            inst_sem_mask[inst_sem_mask > 0] += cur
            inst_pred[inst_sem_mask > 0] = 0
            inst_pred += inst_sem_mask
            cur += len(np.unique(inst_sem_mask))
            sem_pred[inst_sem_mask > 0] = sem_id

        return sem_pred, inst_pred

    # def postprocess_dir(self, sem_pred, dir_pred=None):
    #     """model free post-process for both instance-level & semantic-level."""
    #     # fill instance holes
    #     sem_pred = sem_pred.copy()
    #     sem_pred[sem_pred == self.num_classes] = 0
    #     bin_sem_pred = binary_fill_holes(bin_sem_pred)
    #     # remove small instance
    #     bin_sem_pred = remove_small_objects(bin_sem_pred, 5)
    #     bin_sem_pred = bin_sem_pred.astype(np.uint8)

    #     sem_id_list = list(np.unique(sem_pred))
    #     sem_canvas = np.zeros_like(sem_pred).astype(np.uint8)
    #     for sem_id in sem_id_list:
    #         # 0 is background semantic class.
    #         if sem_id == 0:
    #             continue
    #         sem_id_mask = sem_pred == sem_id
    #         # fill instance holes
    #         sem_id_mask = binary_fill_holes(sem_id_mask)
    #         # remove small instance
    #         sem_id_mask = remove_small_objects(sem_id_mask, 20)
    #         sem_id_mask_dila = morphology.dilation(sem_id_mask, selem=morphology.disk(2))
    #         sem_canvas[sem_id_mask_dila > 0] = sem_id
    #     sem_pred = sem_canvas

    #     bin_sem_pred, bound = mudslide_watershed(bin_sem_pred, dir_pred, sem_pred > 0)

    #     bin_sem_pred = remove_small_objects(bin_sem_pred, 20)
    #     inst_pred = measure.label(bin_sem_pred, connectivity=1)
    #     inst_pred = align_foreground(inst_pred, sem_pred > 0, 20)

    #     return sem_pred, inst_pred

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
        sem_logit_list = []
        dir_logit_list = []
        point_logit_list = []
        img_ = img
        for rotate_degree in self.rotate_degrees:
            for flip_direction in self.flip_directions:
                img = self.tta_transform(img_, rotate_degree, flip_direction)

                # inference
                if self.test_cfg.mode == 'split':
                    sem_logit, dir_logit, point_logit = self.split_inference(img, meta, rescale)
                else:
                    sem_logit, dir_logit, point_logit = self.whole_inference(img, meta, rescale)

                sem_logit = self.reverse_tta_transform(sem_logit, rotate_degree, flip_direction)
                dir_logit = self.reverse_tta_transform(dir_logit, rotate_degree, flip_direction)
                point_logit = self.reverse_tta_transform(point_logit, rotate_degree, flip_direction)

                sem_logit = F.softmax(sem_logit, dim=1)
                dir_logit = F.softmax(dir_logit, dim=1)

                sem_logit_list.append(sem_logit)
                dir_logit_list.append(dir_logit)
                point_logit_list.append(point_logit)

        sem_logit = sum(sem_logit_list) / len(sem_logit_list)
        point_logit = sum(point_logit_list) / len(point_logit_list)

        if rescale:
            sem_logit = resize(sem_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            point_logit = resize(point_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)

        dd_map_list = []
        dir_map_list = []
        for dir_logit in dir_logit_list:
            if rescale:
                dir_logit = resize(dir_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            dir_logit[:, 0] = dir_logit[:, 0] * sem_logit[:, 0]
            dir_map = torch.argmax(dir_logit, dim=1)
            dd_map = generate_direction_differential_map(dir_map, self.num_angles + 1)
            dir_map_list.append(dir_map)
            dd_map_list.append(dd_map)

        dd_map = sum(dd_map_list) / len(dd_map_list)

        if self.test_cfg.get('if_ddm', False):
            sem_logit = self._ddm_enhencement(sem_logit, dd_map, point_logit)

        return sem_logit, dir_map_list[0]

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

        sem_logit = torch.zeros((B, self.num_classes + 1, H1, W1), dtype=img.dtype, device=img.device)
        dir_logit = torch.zeros((B, self.num_angles + 1, H1, W1), dtype=img.dtype, device=img.device)
        point_logit = torch.zeros((B, 1, H1, W1), dtype=img.dtype, device=img.device)
        for i in range(0, H1 - overlap_size, window_size - overlap_size):
            r_end = i + window_size if i + window_size < H1 else H1
            ind1_s = i + overlap_size // 2 if i > 0 else 0
            ind1_e = (i + window_size - overlap_size // 2 if i + window_size < H1 else H1)
            for j in range(0, W1 - overlap_size, window_size - overlap_size):
                c_end = j + window_size if j + window_size < W1 else W1

                img_patch = img_canvas[:, :, i:r_end, j:c_end]
                sem_patch, dir_patch, point_patch = self.calculate(img_patch)

                ind2_s = j + overlap_size // 2 if j > 0 else 0
                ind2_e = (j + window_size - overlap_size // 2 if j + window_size < W1 else W1)
                sem_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = sem_patch[:, :, ind1_s - i:ind1_e - i,
                                                                          ind2_s - j:ind2_e - j]
                dir_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = dir_patch[:, :, ind1_s - i:ind1_e - i,
                                                                          ind2_s - j:ind2_e - j]
                point_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = point_patch[:, :, ind1_s - i:ind1_e - i,
                                                                              ind2_s - j:ind2_e - j]

        sem_logit = sem_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
        dir_logit = dir_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
        point_logit = point_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]

        return sem_logit, dir_logit, point_logit

    def whole_inference(self, img, meta, rescale):
        """Inference with full image."""

        sem_logit, dir_logit, point_logit = self.calculate(img)

        return sem_logit, dir_logit, point_logit

    def _sem_loss(self, sem_logit, sem_gt, weight_map=None):
        sem_loss = {}
        sem_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        sem_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=self.num_classes + 1)
        # Assign weight map for each pixel position
        sem_ce_loss = sem_ce_loss_calculator(sem_logit, sem_gt)
        if weight_map is not None:
            sem_ce_loss *= weight_map[:, 0]
        sem_ce_loss = torch.mean(sem_ce_loss)
        sem_dice_loss = sem_dice_loss_calculator(sem_logit, sem_gt)
        # loss weight
        alpha = 1
        beta = 1
        sem_loss['sem_ce_loss'] = alpha * sem_ce_loss
        sem_loss['sem_dice_loss'] = beta * sem_dice_loss

        return sem_loss

    def _dir_loss(self, dir_logit, dir_gt, weight_map=None):
        dir_loss = {}
        dir_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        dir_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=self.num_angles + 1)
        # Assign weight map for each pixel position
        dir_ce_loss = dir_ce_loss_calculator(dir_logit, dir_gt)
        if weight_map is not None:
            dir_ce_loss *= weight_map[:, 0]
        dir_ce_loss = torch.mean(dir_ce_loss)
        dir_dice_loss = dir_dice_loss_calculator(dir_logit, dir_gt)
        # loss weight
        alpha = 1
        beta = 1
        dir_loss['dir_ce_loss'] = alpha * dir_ce_loss
        dir_loss['dir_dice_loss'] = beta * dir_dice_loss

        return dir_loss

    def _point_loss(self, point_logit, point_gt):
        point_loss = {}
        point_mse_loss_calculator = nn.MSELoss()
        point_mse_loss = point_mse_loss_calculator(point_logit, point_gt)
        # loss weight
        alpha = 1
        point_loss['point_mse_loss'] = alpha * point_mse_loss

        return point_loss

    def _training_metric(self, sem_logit, dir_logit, point_logit, sem_gt, dir_gt, point_gt):
        wrap_dict = {}
        # detach these training variable to avoid gradient noise.
        clean_sem_logit = sem_logit.clone().detach()
        clean_sem_gt = sem_gt.clone().detach()
        clean_dir_logit = dir_logit.clone().detach()
        clean_dir_gt = dir_gt.clone().detach()

        wrap_dict['sem_mdice'] = mdice(clean_sem_logit, clean_sem_gt, self.num_classes)
        wrap_dict['dir_mdice'] = mdice(clean_dir_logit, clean_dir_gt, self.num_angles + 1)

        wrap_dict['sem_tdice'] = tdice(clean_sem_logit, clean_sem_gt, self.num_classes)
        wrap_dict['dir_tdice'] = tdice(clean_dir_logit, clean_dir_gt, self.num_angles + 1)

        # NOTE: training aji calculation metric calculate (This will be deprecated.)
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

    @classmethod
    def _ddm_enhencement(self, sem_logit, dd_map, point_logit):
        # using point map to remove center point direction differential map
        point_logit = point_logit[:, 0, :, :]
        point_map = (point_logit / torch.max(point_logit)) > 0.2
        # point_logit = point_logit - torch.min(point_logit) / (torch.max(point_logit) - torch.min(point_logit))

        # mask out some redundant direction differential
        dd_map = dd_map - (dd_map * point_map)

        # using direction differential map to enhance edge
        sem_logit[:, -1, :, :] = (sem_logit[:, -1, :, :] + dd_map) * (1 + dd_map)

        return sem_logit
