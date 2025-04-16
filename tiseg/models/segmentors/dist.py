"""
Modified from Naylor code at https://github.com/PeterJackNaylor/DRFNS/blob/master/src_RealData/postproc/postprocessing.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import morphology as morph
from skimage import measure
from mmcv.cnn import ConvModule

from tiseg.utils import resize
from ..builder import SEGMENTORS
from ..losses import BatchMultiClassDiceLoss
from .base import BaseSegmentor


def conv1x1(in_dims, out_dims, norm_cfg=None, act_cfg=None):
    return ConvModule(in_dims, out_dims, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)


def conv3x3(in_dims, out_dims, norm_cfg=None, act_cfg=None):
    return ConvModule(in_dims, out_dims, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)


def conv(in_dims, out_dims, kernel, norm_cfg=None, act_cfg=None):
    return ConvModule(in_dims, out_dims, kernel, 1, (kernel - 1) // 2, norm_cfg=norm_cfg, act_cfg=act_cfg)


def prepare_prob(prob_map, convertuint8=True, inverse=True):
    """
    Prepares the prob image for post-processing, it can convert from
    float -> to uint8 and it can inverse it if needed.
    """
    if convertuint8:
        prob_map = prob_map.astype(np.uint8)
    if inverse:
        prob_map = 255 - prob_map
    return prob_map


def H_reconstruction_erosion(prob_img, h):
    """
    Performs a H minimma reconstruction via an erosion method.
    """

    def making_top_mask(x, lamb=h):
        return min(255, x + lamb)

    f = np.vectorize(making_top_mask)
    shift_prob_img = f(prob_img)

    seed = shift_prob_img
    mask = prob_img
    recons = morph.reconstruction(seed, mask, method='erosion').astype(np.dtype('ubyte'))
    return recons


def find_maxima(img, convertuint8=False, inverse=False, mask=None):
    """
    Finds all local maxima from 2D image.
    """
    img = prepare_prob(img, convertuint8=convertuint8, inverse=inverse)
    recons = H_reconstruction_erosion(img, 1)
    if mask is None:
        return recons - img
    else:
        res = recons - img
        res[mask == 0] = 0
        return res


def get_contours(img, radius=2):
    """
    Returns only the contours of the image.
    The image has to be a binary image
    """
    img[img > 0] = 1
    return morph.dilation(img, morph.disk(radius)) - morph.erosion(img, morph.disk(radius))


def generate_wsl(ws):
    """
    Generates watershed line that correspond to areas of
    touching objects.
    """
    se = morph.square(3)
    ero = ws.copy()
    ero[ero == 0] = ero.max() + 1
    ero = morph.erosion(ero, se)
    ero[ws == 0] = 0

    grad = morph.dilation(ws, se) - ero
    grad[ws == 0] = 0
    grad[grad > 0] = 255
    grad = grad.astype(np.uint8)
    return grad


def arrange_label(mat):
    """
    Arrange label image as to effectively put background to 0.
    """
    val, counts = np.unique(mat, return_counts=True)
    background_val = val[np.argmax(counts)]
    mat = measure.label(mat, background=background_val)
    if np.min(mat) < 0:
        mat += np.min(mat)
        mat = arrange_label(mat)
    return mat


def dynamic_watershed_alias(p_img, lamb, p_thresh=0.5):
    """
    Applies our dynamic watershed to 2D prob/dist image.
    """
    b_img = (p_img > p_thresh) + 0
    Probs_inv = prepare_prob(p_img)

    Hrecons = H_reconstruction_erosion(Probs_inv, lamb)
    markers_Probs_inv = find_maxima(Hrecons, mask=b_img)
    markers_Probs_inv = measure.label(markers_Probs_inv)
    ws_labels = morph.watershed(Hrecons, markers_Probs_inv, mask=b_img)
    arranged_label = arrange_label(ws_labels)
    wsl = generate_wsl(arranged_label)
    arranged_label[wsl > 0] = 0

    return arranged_label


@SEGMENTORS.register_module()
class DIST(BaseSegmentor):
    """Implementation of `Segmentation of Nuclei in Histopathology Images by Deep Regression of the Distance Map`.
    """

    def __init__(self, num_classes, train_cfg, test_cfg):
        super(DIST, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_classes = num_classes

        self.stage1 = nn.Sequential(
            conv3x3(3, 32, dict(type='BN'), dict(type='ReLU')),
            conv3x3(32, 32, dict(type='BN'), dict(type='ReLU')),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.stage2 = nn.Sequential(
            conv3x3(32, 64, dict(type='BN'), dict(type='ReLU')),
            conv3x3(64, 64, dict(type='BN'), dict(type='ReLU')),
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.stage3 = nn.Sequential(
            conv3x3(64, 128, dict(type='BN'), dict(type='ReLU')),
            conv3x3(128, 128, dict(type='BN'), dict(type='ReLU')),
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.stage4 = nn.Sequential(
            conv3x3(128, 256, dict(type='BN'), dict(type='ReLU')),
            conv3x3(256, 256, dict(type='BN'), dict(type='ReLU')),
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        self.stage5 = nn.Sequential(
            conv3x3(256, 512, dict(type='BN'), dict(type='ReLU')),
            conv3x3(512, 512, dict(type='BN'), dict(type='ReLU')),
        )

        self.up_conv4 = nn.Sequential(
            conv3x3(512, 256, dict(type='BN'), dict(type='ReLU')),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )
        self.up_stage4 = nn.Sequential(
            conv3x3(512, 256, dict(type='BN'), dict(type='ReLU')),
            conv3x3(256, 256, dict(type='BN'), dict(type='ReLU')),
        )

        self.up_conv3 = nn.Sequential(
            conv3x3(256, 128, dict(type='BN'), dict(type='ReLU')),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )
        self.up_stage3 = nn.Sequential(
            conv3x3(256, 128, dict(type='BN'), dict(type='ReLU')),
            conv3x3(128, 128, dict(type='BN'), dict(type='ReLU')),
        )

        self.up_conv2 = nn.Sequential(
            conv3x3(128, 64, dict(type='BN'), dict(type='ReLU')),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )
        self.up_stage2 = nn.Sequential(
            conv3x3(128, 64, dict(type='BN'), dict(type='ReLU')),
            conv3x3(64, 64, dict(type='BN'), dict(type='ReLU')),
        )

        self.up_conv1 = nn.Sequential(
            conv3x3(64, 32, dict(type='BN'), dict(type='ReLU')),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )
        self.up_stage1 = nn.Sequential(
            conv3x3(64, 32, dict(type='BN'), dict(type='ReLU')),
            conv3x3(32, 32, dict(type='BN'), dict(type='ReLU')),
        )

        self.sem_head = nn.Conv2d(32, self.num_classes, 1)
        self.dist_head = nn.Conv2d(32, 1, 1)

    def calculate(self, img):
        x1 = self.stage1(img)
        p1 = self.pool1(x1)

        x2 = self.stage2(p1)
        p2 = self.pool2(x2)

        x3 = self.stage3(p2)
        p3 = self.pool3(x3)

        x4 = self.stage4(p3)
        p4 = self.pool4(x4)

        x5 = self.stage5(p4)

        x5_up = self.up_conv4(x5)
        c4 = torch.cat([x4, x5_up], dim=1)
        u4 = self.up_stage4(c4)

        u4_up = self.up_conv3(u4)
        c3 = torch.cat([x3, u4_up], dim=1)
        u3 = self.up_stage3(c3)

        u3_up = self.up_conv2(u3)
        c2 = torch.cat([x2, u3_up], dim=1)
        u2 = self.up_stage2(c2)

        u2_up = self.up_conv1(u2)
        c1 = torch.cat([x1, u2_up], dim=1)
        u1 = self.up_stage1(c1)

        sem_logit = self.sem_head(u1)
        dist_logit = self.dist_head(u1)

        return sem_logit, dist_logit

    def forward(self, data, label=None, metas=None, **kwargs):
        """detectron2 style forward functions. Segmentor can be see as meta_arch of detectron2.
        """
        if self.training:
            sem_logit, dist_logit = self.calculate(data['img'])
            assert label is not None
            sem_label = label['sem_gt']
            dist_label = label['dist_gt']
            loss = dict()
            sem_label = sem_label.squeeze(1)
            sem_loss = self._sem_loss(sem_logit, sem_label)
            loss.update(sem_loss)
            dist_loss = self._dist_loss(dist_logit, dist_label)
            loss.update(dist_loss)
            return loss
        else:
            assert metas is not None
            # NOTE: only support batch size = 1 now.
            sem_logit, dist_logit = self.inference(data['img'], metas[0], True)
            sem_pred = sem_logit.argmax(dim=1)
            sem_pred = sem_pred.cpu().numpy()[0]
            dist_logit = dist_logit.cpu().numpy()[0][0]
            sem_pred, inst_pred = self.postprocess(sem_pred, dist_logit)
            # unravel batch dim
            ret_list = []
            ret_list.append({'sem_pred': sem_pred, 'inst_pred': inst_pred})
            return ret_list

    def postprocess(self, sem_pred, dist_logit):
        dist_logit = np.copy(dist_logit)
        dist_logit[dist_logit > 255] = 255
        dist_logit[dist_logit < 0] = 0
        dist_logit = dist_logit.astype('int32')
        # lamb is p1 and p_thresh is p2 in the paper, DIST param
        inst_pred = dynamic_watershed_alias(dist_logit, 0.0, 0.5)
        # sem_pred = (inst_pred > 0).astype(np.uint8)

        return sem_pred, inst_pred

    def _sem_loss(self, sem_logit, sem_label):
        """calculate mask branch loss."""
        sem_loss = {}
        sem_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        sem_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=self.num_classes)
        # Assign weight map for each pixel position
        # mask_loss *= weight_map
        sem_ce_loss = torch.mean(sem_ce_loss_calculator(sem_logit, sem_label.long()))
        sem_dice_loss = sem_dice_loss_calculator(sem_logit, sem_label.long())
        # loss weight
        alpha = 5
        beta = 0.5
        sem_loss['sem_ce_loss'] = alpha * sem_ce_loss
        sem_loss['sem_dice_loss'] = beta * sem_dice_loss

        return sem_loss

    def _dist_loss(self, dist_logit, dist_label):
        """calculate mask branch loss."""
        mask_loss = {}
        mask_mse_loss_calculator = nn.MSELoss()
        # Assign weight map for each pixel position
        # mask_loss *= weight_map
        mask_mse_loss = mask_mse_loss_calculator(dist_logit, dist_label)
        # loss weight
        alpha = 1
        mask_loss['dist_mse_loss'] = alpha * mask_mse_loss

        return mask_loss

    # NOTE: old style split inference
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

        sem_logit = torch.zeros((B, self.num_classes, H1, W1), dtype=img.dtype, device=img.device)
        dist_logit = torch.zeros((B, 1, H1, W1), dtype=img.dtype, device=img.device)
        for i in range(0, H1 - overlap_size, window_size - overlap_size):
            r_end = i + window_size if i + window_size < H1 else H1
            ind1_s = i + overlap_size // 2 if i > 0 else 0
            ind1_e = (i + window_size - overlap_size // 2 if i + window_size < H1 else H1)
            for j in range(0, W1 - overlap_size, window_size - overlap_size):
                c_end = j + window_size if j + window_size < W1 else W1

                img_patch = img_canvas[:, :, i:r_end, j:c_end]
                sem_patch, dist_patch = self.calculate(img_patch)

                ind2_s = j + overlap_size // 2 if j > 0 else 0
                ind2_e = (j + window_size - overlap_size // 2 if j + window_size < W1 else W1)
                sem_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = sem_patch[:, :, ind1_s - i:ind1_e - i,
                                                                          ind2_s - j:ind2_e - j]
                dist_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = dist_patch[:, :, ind1_s - i:ind1_e - i,
                                                                            ind2_s - j:ind2_e - j]

        sem_logit = sem_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
        dist_logit = dist_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
        return sem_logit, dist_logit

    def whole_inference(self, img, meta, rescale):
        """Inference with full image."""

        sem_logit, dist_logit = self.calculate(img)

        return sem_logit, dist_logit

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
        dist_logit_list = []
        img_ = img
        for rotate_degree in self.rotate_degrees:
            for flip_direction in self.flip_directions:
                img = self.tta_transform(img_, rotate_degree, flip_direction)

                # inference patch or whole img
                if self.test_cfg.mode == 'split':
                    sem_logit, dist_logit = self.split_inference(img, meta, rescale)
                else:
                    sem_logit, dist_logit = self.whole_inference(img, meta, rescale)

                sem_logit = self.reverse_tta_transform(sem_logit, rotate_degree, flip_direction)
                dist_logit = self.reverse_tta_transform(dist_logit, rotate_degree, flip_direction)
                sem_logit = F.softmax(sem_logit, dim=1)

                sem_logit_list.append(sem_logit)
                dist_logit_list.append(dist_logit)

        sem_logit = sum(sem_logit_list) / len(sem_logit_list)
        dist_logit = sum(dist_logit_list) / len(dist_logit_list)

        if rescale:
            sem_logit = resize(sem_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)
            dist_logit = resize(dist_logit, size=meta['ori_hw'], mode='bilinear', align_corners=False)

        return sem_logit, dist_logit
