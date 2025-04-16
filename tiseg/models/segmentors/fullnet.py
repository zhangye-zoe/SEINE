"""
Modified from huiqu code at https://github.com/huiqu18/FullNet-varCE/blob/master/FullNet.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import morphology, measure
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes

from ..builder import SEGMENTORS
from ..losses import BatchMultiClassDiceLoss
from .base import BaseSegmentor


class ConvLayer(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(ConvLayer, self).__init__()
        self.add_module(
            'conv',
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
                groups=groups))
        self.add_module('relu', nn.LeakyReLU(inplace=True))
        self.add_module('bn', nn.BatchNorm2d(out_channels))


class BasicLayer(nn.Sequential):

    def __init__(self, in_channels, growth_rate, drop_rate, dilation=1):
        super(BasicLayer, self).__init__()
        self.conv = ConvLayer(in_channels, growth_rate, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv(x)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckLayer(nn.Sequential):

    def __init__(self, in_channels, growth_rate, drop_rate, dilation=1):
        super(BottleneckLayer, self).__init__()

        inter_planes = growth_rate * 4
        self.conv1 = ConvLayer(in_channels, inter_planes, kernel_size=1, padding=0)
        self.conv2 = ConvLayer(inter_planes, growth_rate, kernel_size=3, padding=dilation, dilation=dilation)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


class DenseBlock(nn.Sequential):

    def __init__(self, in_channels, growth_rate, drop_rate, layer_type, dilations):
        super(DenseBlock, self).__init__()
        for i in range(len(dilations)):
            layer = layer_type(in_channels + i * growth_rate, growth_rate, drop_rate, dilations[i])
            self.add_module('denselayer{:d}'.format(i + 1), layer)


def choose_hybrid_dilations(n_layers, dilation_schedule, is_hybrid):
    import numpy as np
    # key: (dilation, n_layers)
    HD_dict = {
        (1, 4): [1, 1, 1, 1],
        (2, 4): [1, 2, 3, 2],
        (4, 4): [1, 2, 5, 9],
        (8, 4): [3, 7, 10, 13],
        (16, 4): [13, 15, 17, 19],
        (1, 6): [1, 1, 1, 1, 1, 1],
        (2, 6): [1, 2, 3, 1, 2, 3],
        (4, 6): [1, 2, 3, 5, 6, 7],
        (8, 6): [2, 5, 7, 9, 11, 14],
        (16, 6): [10, 13, 16, 17, 19, 21]
    }

    dilation_list = np.zeros((len(dilation_schedule), n_layers), dtype=np.int32)

    for i in range(len(dilation_schedule)):
        dilation = dilation_schedule[i]
        if is_hybrid:
            dilation_list[i] = HD_dict[(dilation, n_layers)]
        else:
            dilation_list[i] = [dilation for k in range(n_layers)]

    return dilation_list


@SEGMENTORS.register_module()
class FullNet(BaseSegmentor):

    def __init__(self, num_classes, train_cfg, test_cfg):
        super(FullNet, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_classes = num_classes

        layer_type = BasicLayer

        n_layers = 6
        in_channels = 24
        dilations = (1, 2, 4, 8, 16, 4, 1)
        growth_rate = 24
        compress_ratio = 0.5
        drop_rate = 0.1

        # 1st conv before any dense block
        self.conv1 = ConvLayer(3, 24, kernel_size=3, padding=1)

        self.blocks = nn.Sequential()
        n_blocks = len(dilations)

        dilation_list = choose_hybrid_dilations(n_layers=n_layers, dilation_schedule=dilations, is_hybrid=True)

        for i in range(n_blocks):  # no trans in last block
            block = DenseBlock(in_channels, growth_rate, drop_rate, layer_type, dilation_list[i])
            self.blocks.add_module('block%d' % (i + 1), block)
            num_trans_in = int(in_channels + n_layers * growth_rate)
            num_trans_out = int(math.floor(num_trans_in * compress_ratio))
            trans = ConvLayer(num_trans_in, num_trans_out, kernel_size=1, padding=0)
            self.blocks.add_module('trans%d' % (i + 1), trans)
            in_channels = num_trans_out

        # final conv
        self.conv2 = nn.Conv2d(in_channels, num_classes + 1, kernel_size=3, stride=1, padding=1, bias=False)
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def calculate(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)

        return x

    def forward(self, data, label=None, metas=None, **kwargs):
        """detectron2 style forward functions. Segmentor can be see as meta_arch of detectron2.
        """
        if self.training:
            sem_logit = self.calculate(data['img'])
            assert label is not None
            sem_label = label['sem_gt_w_bound']
            loss = dict()
            sem_label = sem_label.squeeze(1)
            sem_loss = self._sem_loss(sem_logit, sem_label)
            loss.update(sem_loss)
            # calculate training metric
            training_metric_dict = self._training_metric(sem_logit, sem_label)
            loss.update(training_metric_dict)
            return loss
        else:
            assert metas is not None
            # NOTE: only support batch size = 1 now.
            sem_logit = self.inference(data['img'], metas[0], True)
            sem_pred = sem_logit.argmax(dim=1)
            # Extract inside class
            sem_pred = sem_pred.cpu().numpy()[0]
            sem_pred, inst_pred = self.postprocess(sem_pred)
            # unravel batch dim
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

    def _sem_loss(self, sem_logit, sem_gt):
        """calculate mask branch loss."""
        sem_loss = {}
        sem_ce_loss_calculator = nn.CrossEntropyLoss(reduction='none')
        sem_dice_loss_calculator = BatchMultiClassDiceLoss(num_classes=self.num_classes + 1)
        # Assign weight map for each pixel position
        sem_ce_loss = torch.mean(sem_ce_loss_calculator(sem_logit, sem_gt))
        sem_dice_loss = sem_dice_loss_calculator(sem_logit, sem_gt)
        # loss weight
        alpha = 5
        beta = 0.5
        sem_loss['sem_ce_loss'] = alpha * sem_ce_loss
        sem_loss['sem_dice_loss'] = beta * sem_dice_loss

        return sem_loss

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
        for i in range(0, H1 - overlap_size, window_size - overlap_size):
            r_end = i + window_size if i + window_size < H1 else H1
            ind1_s = i + overlap_size // 2 if i > 0 else 0
            ind1_e = (i + window_size - overlap_size // 2 if i + window_size < H1 else H1)
            for j in range(0, W1 - overlap_size, window_size - overlap_size):
                c_end = j + window_size if j + window_size < W1 else W1

                img_patch = img_canvas[:, :, i:r_end, j:c_end]
                sem_patch = self.calculate(img_patch)

                ind2_s = j + overlap_size // 2 if j > 0 else 0
                ind2_e = (j + window_size - overlap_size // 2 if j + window_size < W1 else W1)
                sem_logit[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = sem_patch[:, :, ind1_s - i:ind1_e - i,
                                                                          ind2_s - j:ind2_e - j]

        sem_logit = sem_logit[:, :, (H1 - H) // 2:(H1 - H) // 2 + H, (W1 - W) // 2:(W1 - W) // 2 + W]
        return sem_logit
