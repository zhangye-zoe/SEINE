import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
from mmcv.cnn import ConvModule
from mmcv.cnn import build_activation_layer
from .unet_head import UNetLayer, conv1x1, conv3x3, transconv4x4
import numpy as np


class SGAHead(nn.Module):
    """
    edge guide segmentation head.

    """

    def __init__(self, 
                num_classes=None,
                bottom_in_dim=512,
                skip_in_dims=[64,128,256,512,512],
                stage_dims=[16,32,64,128,256],
                dgm_dims=16,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')):
        super().__init__()
        self.num_classes = num_classes
        self.bottom_in_dim = bottom_in_dim
        self.skip_in_dims = skip_in_dims
        self.stage_dims = stage_dims
        self.dgm_dims=dgm_dims
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.edge_decode_layers = nn.ModuleList()
        self.mask_decode_layers = nn.ModuleList()

        self.attn_layers1 = nn.ModuleList()
        self.attn_layers2 = nn.ModuleList()


        self.mask_conv1 = nn.Conv2d(128, 64, 1, 1)
        self.mask_conv2 = nn.Conv2d(64, 32, 1, 1)
        self.mask_conv3 = nn.Conv2d(32, 16, 1, 1)
        self.mask_conv4 = nn.Conv2d(16, 16, 1, 1)

        # self.mask_conv1 = RU(128, 64)
        # self.mask_conv2 = RU(64, 32)
        # self.mask_conv3 = RU(32, 16)
        # self.mask_conv4 = RU(16, 16)

        self.edge_conv1 = RU(128, 64)
        self.edge_conv2 = RU(64, 32)
        self.edge_conv3 = RU(32, 16)
        self.edge_conv4 = RU(16, 16)

        self.mask_out_layer1 = nn.Conv2d(64,3,1,1)
        self.mask_out_layer2 = nn.Conv2d(32,3,1,1)
        self.mask_out_layer3 = nn.Conv2d(16,3,1,1)
        self.mask_out_layer4 = nn.Conv2d(16,3,1,1)

        self.edge_out_layer1 = nn.Conv2d(64,1,1,1)
        self.edge_out_layer2 = nn.Conv2d(32,1,1,1)
        self.edge_out_layer3 = nn.Conv2d(16,1,1,1)
        self.edge_out_layer4 = nn.Conv2d(16,1,1,1)

        self.ru1 = RU(16, 16, norm_cfg, act_cfg)
        self.ru2 = RU(16, 16, norm_cfg, act_cfg)
        self.pos_mask_layer = nn.Conv2d(16,1,1,1)
        self.pos_edge_layer = nn.Conv2d(16,1,1,1)

        num_layers = len(self.skip_in_dims)

        for idx in range(num_layers - 1, -1, -1):
            
            if idx == num_layers - 1:
                self.edge_decode_layers.append(
                    UNetLayer(self.bottom_in_dim*2, self.skip_in_dims[idx], self.stage_dims[idx], 2, norm_cfg, act_cfg))
                # self.edge_decode_layers.append(
                #     UNetLayer(self.bottom_in_dim, self.skip_in_dims[idx], self.stage_dims[idx], 2, norm_cfg, act_cfg))
                self.mask_decode_layers.append(
                    UNetLayer(self.bottom_in_dim, self.skip_in_dims[idx], self.stage_dims[idx], 2, norm_cfg, act_cfg))

                self.attn_layers1.append(EdgeGAttn(self.stage_dims[idx], self.stage_dims[idx]))
                self.attn_layers2.append(EdgeGAttn(self.stage_dims[idx], self.stage_dims[idx]))

            else:
            
                self.edge_decode_layers.append(
                    UNetLayer(self.stage_dims[idx + 1]*2, self.skip_in_dims[idx], self.stage_dims[idx], 2, norm_cfg,
                            act_cfg))
                
                # self.edge_decode_layers.append(
                #     UNetLayer(self.stage_dims[idx + 1], self.skip_in_dims[idx], self.stage_dims[idx], 2, norm_cfg,
                #             act_cfg))
                self.mask_decode_layers.append(
                    UNetLayer(self.stage_dims[idx + 1], self.skip_in_dims[idx], self.stage_dims[idx], 2, norm_cfg,
                            act_cfg))
                if idx != 0:
                    self.attn_layers1.append(EdgeGAttn(int(self.stage_dims[idx]/2), int(self.stage_dims[idx]/2)))
                    self.attn_layers2.append(EdgeGAttn(int(self.stage_dims[idx]/2), int(self.stage_dims[idx]/2)))
                if idx == 0:
                    self.attn_layers1.append(EdgeGAttn(self.stage_dims[idx], self.stage_dims[idx]))
                    self.attn_layers2.append(EdgeGAttn(self.stage_dims[idx], self.stage_dims[idx]))


    def forward(self, bottom_input, skip_inputs):
        edge_feature = bottom_input
        mask_feature = bottom_input
        skips = skip_inputs[::-1]
        edge_decode_layers = self.edge_decode_layers
        mask_decode_layers = self.mask_decode_layers
        attn_layers1 = self.attn_layers1
        attn_layers2 = self.attn_layers2
        all_mask_feature = []
        all_edge_feature = []

        # print("att layer1", attn_layers1)
        # print("att layer2", attn_layers2)

        for i, v in enumerate(zip(skips, edge_decode_layers, mask_decode_layers, attn_layers1, attn_layers2)):
            skip, edge_decode_stage, mask_decode_stage, attn_stage1, attn_stage2 = v[0],v[1],v[2],v[3],v[4]
            
            # print('edge feature', edge_feature.shape)
            # print('mask fature', mask_feature.shape)
            # print('=' * 100)
            
            edge_feature = torch.concat((edge_feature, mask_feature), axis=1)
            
            edge_feature = edge_decode_stage(edge_feature, skip)
            mask_feature = mask_decode_stage(mask_feature, skip)
            # print(f"{i}", "=" * 50)
            # print("mask feature", mask_feature.shape)
            # print("edge feature", edge_feature.shape)
            

            if i == 0:
                pass
                # out_edge_feature = edge_feature
            else:
                mask_conv_layer = getattr(self, "mask_conv{}".format(i))
                edge_conv_layer = getattr(self, "edge_conv{}".format(i))

                att_mask_feature = mask_conv_layer(mask_feature)
                att_edge_feature = edge_conv_layer(edge_feature)
                # print(f"{i}", "+" * 100)
                # print("After mask feature", mask_feature.shape)
                # print("After edge feature", edge_feature.shape)

                att_mask_feature, att_h1, att_w1 = attn_stage1(att_edge_feature, att_mask_feature)
                # att_mask_feature, att_h2, att_w2 = attn_stage2(att_edge_feature, att_mask_feature)

                # print("attention mask feature", att_mask_feature.shape)
                # print("=" * 100)
                # size = att_mask_feature.shape[-1]
                # np.save(f"z_feature_map_{size}.png", att_mask_feature.cpu().numpy())
                



                mask_out_layer = getattr(self, "mask_out_layer{}".format(i))
                edge_out_layer = getattr(self, "edge_out_layer{}".format(i))
                out_mask_feature = mask_out_layer(att_mask_feature)
                out_edge_feature = edge_out_layer(att_edge_feature)
                all_mask_feature.append(out_mask_feature)
                all_edge_feature.append(out_edge_feature)

                # np.save(f"z_att_h{i}", att_h1.cpu().numpy())
                # np.save(f"z_att_w{i}", att_w1.cpu().numpy())

                # import matplotlib.pyplot as plt
                # plt.imshow(mask_feature.cpu().numpy()[0,1,...])
                # plt.show()
                # plt.savefig("z_mask_feature.png")
                if i == 4:
                #     import matplotlib.pyplot as plt
                    # plt.imshow(out_edge_feature.detach().cpu().numpy()[0,0,...])
                    # plt.show()
                    # plt.savefig("z_feature.png")

                    # print("att_h1", att_h1*1000)
                    # print(att_h1.cpu().numpy())
                    # np.save("z_att_h4", att_h1.cpu().numpy())
                    # np.save("z_att_w4", att_w1.cpu().numpy())
                    # np.save("z_att_h2", att_h2.cpu().numpy())
                    # np.save("z_att_w2", att_w2.cpu().numpy())

                    pos_mask_fea = self.pos_mask_layer(self.ru1(mask_feature))
                    pos_edge_fea = self.pos_edge_layer(self.ru2(edge_feature))

        return all_mask_feature, all_edge_feature, pos_mask_fea, pos_edge_fea



def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

class EdgeGAttn(nn.Module):
    def __init__(self, in_dim, out_dim ):
        super(EdgeGAttn,self).__init__()

        # self.edge_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, padding=1)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)

        # self.mask_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3,padding=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)

        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=3)
        

    def forward(self, edge_feature, mask_feature):
        bs, channel, height, width = edge_feature.size()

        # edge_fea = self.edge_conv1(edge_feature)
        # mask_fea = self.mask_conv1(mask_feature)

        proj_query = self.query_conv(edge_feature)
        # print("proj query", proj_query.shape)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(bs*width,-1,height).permute(0, 2, 1) # (12,5,8)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(bs*height,-1,width).permute(0, 2, 1) # (10,6,8)

        proj_key = self.key_conv(edge_feature)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(bs*width,-1,height) # (12,8,5)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(bs*height,-1,width) # (10,8,6)
        
        proj_value = self.value_conv(mask_feature) 
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(bs*width,-1,height) #(12,64,5)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(bs*height,-1,width) #(10,64,6)
        
        # (2,5,6,5)
        # energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(bs, height, width)).view(bs,width,height,height).permute(0,2,1,3)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)).view(bs,width,height,height).permute(0,2,1,3)
        # (2,5,6,6)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(bs,height,width,width)
        # print("energy H", energy_H.shape)
        # print("energy H", energy_W.shape)
        # print("=" * 100)
        

        # (2,5,6,11)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        # (12,5,5)
        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(bs*width,height,height)
        # (10, 6, 6)
        att_W = concate[:,:,:,height:height+width].contiguous().view(bs*height,width,width)

        # print("H", att_H.shape, att_H)
        # print("W", att_W.shape, att_W)
        # print("=" * 100)

        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(bs,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(bs,height,-1,width).permute(0,2,1,3)
        # print(out_H.size(),out_W.size())

        # v1 = torch.ones_like(proj_value_H)
        # v2 = torch.ones_like(proj_value_W)
        # out_v1 = torch.bmm(v1, att_H.permute(0, 2, 1)).view(bs,width,-1,height).permute(0,2,3,1)
        # out_v2 = torch.bmm(v2, att_W.permute(0, 2, 1)).view(bs,height,-1,width).permute(0,2,1,3)

        # import matplotlib.pyplot as plt 
        # out = out_v1 + out_v2

        # print("attn", att_H[0].detach().cpu().numpy())
        # print("=" * 100)
        # plt.imshow(out[0,0].detach().cpu().numpy())
        # plt.show()
        # plt.savefig("zz_attn.png")
        # print('输入')
        # print("att H", att_H.shape)
        # print("att W", att_W.shape)
        # plt.imshow(att_H[0].detach().cpu().numpy())
        # plt.show()
        # plt.savefig("zz_attn1.png")
        # print('输出')


        return self.gamma*(out_H + out_W) + mask_feature , att_H, att_W


class DGM(nn.Module):
    """Direction-Guided Refinement Module (DGM)

    This module will accept prediction of regular segmentation output. This
    module has three branches:
    (1) Mask Branch;
    (2) Direction Map Branch;
    (3) Point Map Branch;

    When training phrase, these three branches provide mask, direction, point
    supervision, respectively. When testing phrase, direction map and point map
    provide refinement operations.

    Args:
        in_dims (int): The input channels of DGM.
        feed_dims (int): The feedforward channels of DGM.
        num_classes (int): The number of mask semantic classes.
        num_angles (int): The number of angle types. Default: 8
        norm_cfg (dict): The normalize layer config. Default: dict(type='BN')
        act_cfg (dict): The activation layer config. Default: dict(type='ReLU')
    """

    def __init__(self,
                 in_dims,
                 feed_dims,
                 num_classes,
                 num_angles=8,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.in_dims = in_dims
        self.feed_dims = feed_dims
        self.num_classes = num_classes
        self.num_angles = num_angles

        self.mask_feats = RU(self.in_dims, self.feed_dims, norm_cfg, act_cfg)
        # self.dir_feats = RU(self.feed_dims, self.feed_dims, norm_cfg, act_cfg)
        # self.point_feats = RU(self.feed_dims, self.feed_dims, norm_cfg, act_cfg)

        # # Cross Branch Attention
        # self.point_to_dir_attn = AU(1)
        # self.dir_to_mask_attn = AU(self.num_angles + 1)

        # # Prediction Operations
        # self.point_conv = nn.Conv2d(self.feed_dims, 1, kernel_size=1)
        # self.dir_conv = nn.Conv2d(self.feed_dims, self.num_angles + 1, kernel_size=1)
        # self.mask_conv = nn.Conv2d(self.feed_dims, self.num_classes, kernel_size=1)

    def forward(self, x):
        mask_feature = self.mask_feats(x)
        # dir_feature = self.dir_feats(mask_feature)
        # point_feature = self.point_feats(dir_feature)

        # # point branch
        # point_logit = self.point_conv(point_feature)

        # # direction branch
        # dir_feature_with_point_logit = self.point_to_dir_attn(dir_feature, point_logit)
        # dir_logit = self.dir_conv(dir_feature_with_point_logit)

        # # mask branch
        # mask_feature_with_dir_logit = self.dir_to_mask_attn(mask_feature, dir_logit)
        # mask_logit = self.mask_conv(mask_feature_with_dir_logit)
        # print("mask_feature_with_dir_logit", mask_feature_with_dir_logit.shape)
        # print("mask logit", mask_logit.shape)

        # return mask_feature_with_dir_logit, mask_logit, dir_logit, point_logit
        return mask_feature

class RU(nn.Module):
    """Residual Unit.

    Residual Unit comprises of:
    (Conv3x3 + BN + ReLU + Conv3x3 + BN) + Identity + ReLU
    ( . ) stands for residual inside block

    Args:
        in_dims (int): The input channels of Residual Unit.
        out_dims (int): The output channels of Residual Unit.
        norm_cfg (dict): The normalize layer config. Default: dict(type='BN')
        act_cfg (dict): The activation layer config. Default: dict(type='ReLU')
    """

    def __init__(self, in_dims, out_dims, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')):
        super().__init__()

        # NOTE: inplace wise relu can largely save gpu memory cost.
        real_act_cfg = dict()
        real_act_cfg['inplace'] = True
        real_act_cfg.update(act_cfg)

        self.act_layer = build_activation_layer(real_act_cfg)
        self.residual_ops = nn.Sequential(
            conv3x3(in_dims, out_dims, norm_cfg), self.act_layer, conv3x3(out_dims, out_dims, norm_cfg))
        self.identity_ops = nn.Sequential(conv1x1(in_dims, out_dims))

    def forward(self, x):
        ide_value = self.identity_ops(x)
        res_value = self.residual_ops(x)
        out = ide_value + res_value
        return self.act_layer(out)


class AU(nn.Module):
    """Attention Unit.

    This module use (conv1x1 + sigmoid) to generate 0-1 (float) attention mask.

    Args:
        in_dims (int): The input channels of Attention Unit.
        num_masks (int): The number of masks to generate. Default: 1
    """

    def __init__(self, in_dims, num_masks=1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_dims, num_masks, kernel_size=1, bias=False), nn.Sigmoid())

    def forward(self, signal, gate):
        """Using gate to generate attention map and assign the attention map to
        signal."""
        attn_map = self.conv(gate)
        return signal * (1 + attn_map)

    

