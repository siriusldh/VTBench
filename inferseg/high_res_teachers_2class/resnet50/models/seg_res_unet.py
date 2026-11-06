# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_divisible(v, divisor=4, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNPReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, groups=1):
        padding = (kernel_size - 1) // 2 * dilation
        super(ConvBNPReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.PReLU(out_planes)
        )

class EncoderBlock(nn.Module):
    def __init__(self, inp, expanded_depth, oup, kernels, stride):
        super(EncoderBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        for i, kernel in enumerate(kernels):
            branch = []
            if expanded_depth != inp:
                branch.append(ConvBNPReLU(inp, expanded_depth, 1, 1))
            # dw
            branch.append(ConvBNPReLU(expanded_depth, expanded_depth, kernel_size=kernel, stride=stride, groups=expanded_depth))

            layers.append(nn.Sequential(*branch))

        self.branches = nn.ModuleList(layers)
        branch_num = len(kernels)
        # pw-linear
        self.pw = nn.Sequential(nn.Conv2d(expanded_depth*branch_num, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))

    def forward(self, x):
        out_list = [branch(x) for i, branch in enumerate(self.branches)]
        out = self.pw(torch.cat(out_list, dim=1))
        if self.use_res_connect:
            return x + out
        else:
            return out


class Decode_block(nn.Module):
    def __init__(self, low_c, high_c, out_c, kernels):
        super(Decode_block, self).__init__()
        down_dim = _make_divisible(high_c//4)
        # self.conv_low = nn.Sequential(
        #                     nn.Conv2d(low_c, down_dim, kernel_size=1, bias=False),
        #                     nn.BatchNorm2d(down_dim),
        #                     nn.PReLU(down_dim)
        # )
        self.conv_low = EncoderBlock(low_c, low_c, down_dim, [3], 1)

        self.conv_merge = EncoderBlock(high_c+down_dim, high_c+down_dim, out_c, kernels, 1)

    def forward(self, lower_layer, high_layer):
        lower_layer = self.conv_low(lower_layer)
        high_layer = F.interpolate(high_layer, scale_factor=2, mode='bilinear', align_corners=False)
        cat_layer = torch.cat((high_layer, lower_layer), dim=1)
        link_layer = self.conv_merge(cat_layer)
        return link_layer


class UNetCatOs2(nn.Module):
    def __init__(self, selected_chs, out_channels):
        super(UNetCatOs2, self).__init__()

        self.decoder_16 = Decode_block(selected_chs[-2], selected_chs[-1], out_channels*8, [3])
        self.decoder_8  = Decode_block(selected_chs[-3], out_channels*8,   out_channels*4, [3])
        self.decoder_4  = Decode_block(selected_chs[-4], out_channels*4,   out_channels*2, [3])
        self.decoder_2  = Decode_block(selected_chs[-5], out_channels*2,   out_channels,   [3])
        self.fine = EncoderBlock(out_channels, out_channels, out_channels, [3], 1)

        self.channels = [selected_chs[-1], out_channels*8, out_channels*4, out_channels*2, out_channels]

    def forward(self, features):

        c1, c2, c3, c4, c5 = features

        fine_16 = self.decoder_16(c4, c5)
        fine_8 = self.decoder_8(c3, fine_16)
        fine_4 = self.decoder_4(c2, fine_8)
        fine_2 = self.decoder_2(c1, fine_4)
        before_cls = self.fine(fine_2)

        outputs = [c5, fine_16, fine_8, fine_4, before_cls]

        return outputs


def unet_cat_os2(selected_chs, out_channels, has_stage5):
    return UNetCatOs2(selected_chs, out_channels)



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        
        self.mean_vec = (0.485, 0.456, 0.406)
        self.std_vec = (0.229, 0.224, 0.225)
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64

        self.channels = [self.inplanes] + [self.inplanes * block.expansion * t for t in [1, 2, 4, 8]]

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)

        x = self.maxpool(x0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x0, x1, x2, x3, x4]


def _resnet(block, layers, pretrained, pretrained_path):
    model = ResNet(block, layers)
    if pretrained:
        import os
        assert os.path.exists(pretrained_path)
        all_params = {}
        pretrained_dict = torch.load(pretrained_path)
        for k, v in model.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
        # assert len(all_params.keys()) == len(model.state_dict().keys())
        model.load_state_dict(all_params)

    return model


def resnet50(pretrained=False, pretrained_path='./pretrained_model/best_ver_res50_unetcat_256_128.pth'):
    return _resnet(Bottleneck, [3, 4, 6, 3], pretrained, pretrained_path)




class BinarySegmentation(nn.Module):
    def __init__(self, binary_class=2, parsing_class=2, edge_class=2,
                 num_stage=5, decoder_channels=64, has_edge=False, has_parsing=True, refinement=None, refine_chs=4, kd=False,
                 has_ds=False, align_corners=False, output_softmax=False, output_size=[256,256], input_4in=False):
        super(BinarySegmentation, self).__init__()

        self.encoder = resnet50(False)
        self.mean_vec = self.encoder.mean_vec
        self.std_vec = self.encoder.std_vec
        self.encoder_chs = self.encoder.channels

        self.decoder = unet_cat_os2(self.encoder_chs, decoder_channels, num_stage)
        self.decoder_chs = self.decoder.channels

        self.has_edge = has_edge
        self.has_parsing = has_parsing
        self.refinement = refinement
        self.refine_chs = refine_chs
        self.kd = kd
        self.output_softmax = output_softmax
        self.output_size = output_size
        self.input_4in = input_4in

        self.align_corners = align_corners
        print('Align Corners is True' if self.align_corners else 'Align Corners is False')

        # ###########Deeply Supervision##########
        self.has_ds = has_ds
        if self.has_ds:
            self.out_binary_side_x4 = AuSegmentHead(self.decoder_chs[-2], binary_class)
            self.out_binary_side_x8 = AuSegmentHead(self.decoder_chs[-3], binary_class)
            self.out_binary_side_x16 = AuSegmentHead(self.decoder_chs[-4], binary_class)
            self.out_binary_side_x32 = AuSegmentHead(self.decoder_chs[-5], binary_class)

        # #######################################

        if self.has_parsing:
            self.out_conv_parsing = nn.Conv2d(self.decoder_chs[-1], parsing_class, 1)

        # if self.has_edge:
        #     self.out_conv_edge = nn.Conv2d(self.decoder_chs, edge_class, 1)

        self.out_conv_binary = nn.Conv2d(self.decoder_chs[-1], binary_class, 1)

        if self.input_4in:
            print('use 4 channels as input, [rgb,pre_mask]')
            self.pre_acc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(self.encoder_chs[-1], 1, 1), nn.Sigmoid())

        # if self.refinement is not None:
        #     self.refine_module = build_refinement(self.refinement, binary_class, self.refine_chs, binary_class)
        self.init_weights()

    def forward(self, x):

        preds = []

        feats_encoder = self.encoder(x)
        feats_decoder = self.decoder(feats_encoder)

        pred_binary = self.out_conv_binary(feats_decoder[-1])

        # ###########Deeply Supervision##########
        preds_side = []
        if self.has_ds:
            preds_side_x4 = self.out_binary_side_x4(feats_decoder[-2])
            preds_side_x8 = self.out_binary_side_x8(feats_decoder[-3])
            preds_side_x16 = self.out_binary_side_x16(feats_decoder[-4])
            preds_side_x32 = self.out_binary_side_x32(feats_decoder[-5])

            preds_side.append(preds_side_x4)
            preds_side.append(preds_side_x8)
            preds_side.append(preds_side_x16)
            preds_side.append(preds_side_x32)

            # preds_side = [F.interpolate(pred_side, size=self.output_size, mode='bilinear', align_corners=False) for pred_side in preds_side]


        ######################################

        # if pred_binary.size()[2:] != (h, w):
        # print("--- pred_binary.size() ---")
        # print(pred_binary.size())
        # pred_binary = F.interpolate(pred_binary, size=self.output_size, mode='bilinear', align_corners=self.align_corners)

        preds.append(pred_binary)

        if self.has_parsing:
            pred_parsing = self.out_conv_parsing(feats_decoder[-1])
            # if pred_parsing.size()[2:] != (h, w):
            # pred_parsing = F.interpolate(pred_parsing, size=self.output_size, mode='bilinear', align_corners=self.align_corners)

            preds.append(pred_parsing)
        else:
            preds.append(None)

        if self.output_softmax:
            return torch.softmax(preds[0], dim=1)
        else:
            return preds

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

def get_seg_model(cfg, **kwargs):
    model = BinarySegmentation(decoder_channels=64,
                        has_edge=False,
                        has_parsing=True,
                        has_ds=False)
    #model.init_weights(cfg.MODEL.PRETRAINED)

    model_dict = torch.load(cfg.MODEL.PRETRAINED)

    checkpoint_no_module = {}

    for k, v in model_dict.items():
        if k.startswith('module'):
            k = k[7:]
        if k.startswith('model.'):
            k = k[6:]
        checkpoint_no_module[k] = v
    model.load_state_dict(checkpoint_no_module)
    return model


if __name__ == '__main__':
    import numpy as np

    data = torch.rand(1, 3, 512, 512)

    res50_unet_cat = BinarySegmentation(decoder_channels=64,
                        has_edge=False,
                        has_parsing=True,
                        has_ds=False)
    
    pth_path = 'pretrained_model/best_ver_res50_unetcat_256_128.pth'
    model_dict = torch.load(pth_path, map_location=lambda storage, loc: storage)

    checkpoint_no_module = {}

    for k, v in model_dict.items():
        if k.startswith('module'):
            k = k[7:]
        if k.startswith('model.'):
            k = k[6:]
        checkpoint_no_module[k] = v
    res50_unet_cat.load_state_dict(checkpoint_no_module)

    with torch.no_grad():
        # torch.onnx.export(res50_unet_cat, data, "resnet50_unet.onnx")
        out = res50_unet_cat(data)

        print(type(out))
        print(out[1].size())
