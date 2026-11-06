# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch import nn, Tensor


class _Wrapper(nn.Module):

    def __init__(self, impl: nn.Module, train: bool):
        super().__init__()
        self.impl = impl
        self.train_mode = train

    def forward(self, x: Tensor) -> Tensor:
        in_h, in_w = tuple(int(v) for v in x.size()[2:4])
        outs       = self.impl(x)
        hair       = nn.functional.interpolate(outs[0], size=(in_h, in_w),
                                               mode='bilinear',
                                               align_corners=False)
        face       = nn.functional.interpolate(outs[1], size=(in_h, in_w),
                                               mode='bilinear',
                                               align_corners=False)
        out        = nn.functional.interpolate(outs[2], size=(in_h, in_w),
                                               mode='bilinear',
                                               align_corners=False)
        out_refine = nn.functional.interpolate(outs[3], size=(in_h, in_w),
                                               mode='bilinear',
                                               align_corners=False)
        if self.train_mode:
            return out_refine, out, face, hair
        else:
            return [out_refine, face, hair]


def get(train=True):
    from .seg_hrnet_ocr import HighResolutionNet
    from yaml import safe_load
    from os.path import dirname, abspath
    from easydict import EasyDict
    import torch

    with open(dirname(abspath(__file__)) + '/seg_hrnet_ocr.yaml', 'r') as f:
        cfg_dict = safe_load(f)
    cfg = EasyDict(cfg_dict)

    m   = HighResolutionNet(cfg)

    # w   = torch.load('/apdcephfs/share_1307729/shared_info/cooperrfeng/trained_pths/hrocr_512x512_0128.pth',
    #                  map_location='cpu')
    # w2  = {k[6:]: v for k, v in w.items()}
    # info = m.load_state_dict(w2, strict=False)
    # print(info)
    if train:
        m.train()
    else:
        m.eval()

    return _Wrapper(m, train)
