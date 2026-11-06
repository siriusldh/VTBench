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

    def __init__(self, impl: nn.Module):
        super().__init__()
        self.impl = impl

    def forward(self, x: Tensor) -> Tensor:
        in_h, in_w = tuple(int(v) for v in x.size()[2:4])
        outs       = self.impl(x)
        out        = nn.functional.interpolate(outs[0], size=(in_h, in_w),
                                               mode='bilinear',
                                               align_corners=False)
        return out


def get():
    from .seg_res_unet import BinarySegmentation
    from os.path import dirname, abspath
    import torch

    m = BinarySegmentation(decoder_channels=64,
                           has_edge=False,
                           has_parsing=True,
                           has_ds=False)

    w   = torch.load(dirname(abspath(__file__)) + '/../final_state.pth',
                     map_location='cpu')
    w2  = {k[6:]: v for k, v in w.items()}
    m.load_state_dict(w2)
    return _Wrapper(m)
