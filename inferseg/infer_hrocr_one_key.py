import os
import numpy as np
import cv2
import glob

import torchvision.transforms as transforms
import torch
from PIL import Image
import high_res_teachers_2class.hrnet_ocr.models  
import high_res_teachers_2class.resnet101.models  # noqa
from postprocess import SegVideoPostproc2
from tqdm import tqdm
# import pspnet

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_dir', type=str,
                    help='Input image dir')
parser.add_argument('--result_dir', type=str,
                    help='Result mask dir')
parser.add_argument('--output_comp', action='store_true', help='output composite images')
parser.add_argument('--background_bgr', type=int, default=[0,0,0], nargs='+', help='use what bg')
parser.add_argument('--background_img', type=str, default=None, help='use what bg')
parser.add_argument('--output_mask', action='store_true', help='output mask')
parser.add_argument('--mask_mode', type=str, default='float', help='float, binary, binary-255')
parser.add_argument('--guide_filter', action='store_true', help='use guide filter')
parser.add_argument('--ridge_regression', action='store_true', help='use ridge regression')


class EvalNet():
    def __init__(self, model_type, model_path, input_size, num_class=2, tp='Hor'):
        torch.set_grad_enabled(False)
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.test_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # if model_type not in model_type_list:
        #     print(model_type_list)
        #assert model_type in model_type_list
        if model_type== 'res101':
            self.netSeg = high_res_teachers_2class.resnet101.models.get(train=False)
        elif model_type== 'res101_unetsumos2':
            self.netSeg = binary_segmentation.BinarySegmentation(encoder='resnet101', decoder='unet_sum_os2_nc1',
                                                                binary_class=2, parsing_class=2,
                                                                edge_class=2,
                                                                num_stage=5, decoder_channels=128,
                                                                has_edge=False, has_parsing=False,
                                                                refinement=None, kd=False, refine_chs=64,
                                                                has_ds=False)
        elif model_type == 'mbv2_unetsumos2':
            self.netSeg = binary_segmentation.BinarySegmentation(encoder='mobilenetv2_05', decoder='unet_sum_os2_nc1',
                                                                binary_class=2, parsing_class=2,
                                                                edge_class=2,
                                                                num_stage=5, decoder_channels=4,
                                                                has_edge=False, has_parsing=False,
                                                                refinement=None, kd=False, refine_chs=64,
                                                                has_ds=False)
        elif model_type == 'mbv2-1_unetsumos2':
            self.netSeg = binary_segmentation.BinarySegmentation(encoder='mobilenetv2', decoder='unet_sum_os2_nc1',
                                                                binary_class=2, parsing_class=2,
                                                                edge_class=2,
                                                                num_stage=5, decoder_channels=4,
                                                                has_edge=False, has_parsing=False,
                                                                refinement=None, kd=False, refine_chs=64,
                                                                has_ds=False)
        elif model_type == 'mbv2_unet_ms':
            self.netSeg = binary_segmentation.BinarySegmentation_multi(encoder='mobilenetv2_05', decoder='unet_sum_os2_nc1',
                                                                binary_class=2, parsing_class=2,
                                                                edge_class=2,
                                                                num_stage=5, decoder_channels=4,
                                                                has_edge=False, has_parsing=False,
                                                                refinement=None, kd=False, refine_chs=64,
                                                                has_ds=False)
        elif model_type == "hrnet":
            self.netSeg = model_seg_hrnet.HighResolutionNet(output_size=(input_size[1], input_size[0]), output_face=True)
        
        elif model_type == "hrnet_ocr":
            self.netSeg = high_res_teachers_2class.hrnet_ocr.models.get(train=False)
        elif model_type == "hrsinet":
            self.netSeg = model_seg_hrsinet.HighResolutionNet(output_size=(input_size[1], input_size[0]), output_face=False)    
        
        elif model_type == 'res50_unetcat':
            self.netSeg = binary_segmentation.BinarySegmentation(encoder='resnet50', decoder='unet_cat_os2',
                                                                 binary_class=2, parsing_class=2,
                                                                 edge_class=2,
                                                                 num_stage=5,
                                                                 decoder_channels=64,
                                                                 has_edge=False, has_parsing=True,
                                                                 refinement=False, kd=False,
                                                                 refine_chs=False, has_ds=False)
        elif model_type == 'res50_unetsum':
            self.netSeg = binary_segmentation.BinarySegmentation(encoder='resnet50', decoder='unet_sum_os2_nc1',
                                                                 binary_class=2, parsing_class=2,
                                                                 edge_class=2,
                                                                 num_stage=5,
                                                                 decoder_channels=64,
                                                                 has_edge=False, has_parsing=True,
                                                                 refinement=False, kd=False,
                                                                 refine_chs=False, has_ds=False)                                                       
        elif model_type == "eff0_unet":
            self.netSeg = binary_segmentation.BinarySegmentation(encoder='tf_efficientnet_lite0', decoder='unet_sum_os2_nc1',
                                                                binary_class=2, parsing_class=2,
                                                                edge_class=2,
                                                                num_stage=5, decoder_channels=4,
                                                                has_edge=False, has_parsing=False,
                                                                refinement=None, kd=False, refine_chs=64,
                                                                has_ds=False)
        elif model_type == 'resnest50_unetcat':
            self.netSeg = binary_segmentation.BinarySegmentation(encoder='resnest50', decoder='unet_cat_os2',
                                                                binary_class=2, parsing_class=2,
                                                                edge_class=2,
                                                                num_stage=5, decoder_channels=64,
                                                                has_edge=False, has_parsing=True,
                                                                refinement=None, kd=False, refine_chs=64,
                                                                has_ds=False)
        elif model_type == "sinet_hor":
            self.netSeg = binary_segmentation.BinarySegmentation(encoder='sinet_encoder_p2_q8_nosqueeze_noshuffle_160_224', decoder='sinet_decoder_160_224',
                                                                binary_class=2, parsing_class=2,
                                                                edge_class=2,
                                                                num_stage=5, decoder_channels=2,
                                                                has_edge=False, has_parsing=False,
                                                                refinement=None, kd=False, refine_chs=64,
                                                                has_ds=False)
        elif model_type == "sinet_Ver":
            self.netSeg = binary_segmentation.BinarySegmentation(encoder='sinet_encoder_p2_q8_nosqueeze_noshuffle_224_160', decoder='sinet_decoder_224_160',
                                                                binary_class=2, parsing_class=2,
                                                                edge_class=2,
                                                                num_stage=5, decoder_channels=2,
                                                                has_edge=False, has_parsing=False,
                                                                refinement=None, kd=False, refine_chs=64,
                                                                has_ds=False)
        elif model_type == "mbms_unet4_edge":
            self.netSeg = model_seg_mbms.MBMS_unet_4_Edge(output_face=True, output_size=[input_size[1], input_size[0]])
        elif model_type == "mbms_unet4_edge_1":
            self.netSeg = model_seg_mbms.MBMS_unet_4_Edge_1(output_face=True, output_size=[input_size[1], input_size[0]])
        elif model_type == "mbms_unet3":
            self.netSeg = model_seg_mbms.MBMS_unet_3(output_face=True, output_size=[input_size[1], input_size[0]])
        else:
            print('support model arc:')
            raise TypeError("Cannot handle this model type")
        # self.netSeg = model_seg_mbv3.SegMBV3_cx(num_class, output_size)
        pretrained_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        if 'state_dict' in pretrained_dict.keys():
            pretrained_dict = pretrained_dict['state_dict']
        checkpoint_no_module = {}
        for k, v in pretrained_dict.items():
            if k.startswith('module'):
                k = k[7:]
#             if k.startswith('model'):
#                 k = k[6:]
            if k.startswith('model'):
                k = k.replace('model', 'impl')
            checkpoint_no_module[k] = v
        # print(self.netSeg)
        #print(checkpoint_no_module)
        info = self.netSeg.load_state_dict(checkpoint_no_module, strict=False)
        print(info)
        self.input_size = input_size

        self.netSeg.to(self.test_device)
        self.netSeg = self.netSeg.eval()
        self.pre_mask=None
        self.tp = tp

    def __call__(self, im_cv2, pre_ratio=0): 
        h,w,c = im_cv2.shape
        mask_np = None
        mask_soft_np = None
        tp = self.tp
        bgr = cv2.resize(im_cv2, self.input_size, interpolation=cv2.INTER_LINEAR)
        #print(bgr.shape)
        im_pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        # im_pil = Image.fromarray(cv2.resize(np.array(im_pil), self.input_size, interpolation=cv2.INTER_LINEAR))
        pred = self.netSeg(self.test_transform(im_pil).unsqueeze(0).to(self.test_device))
        pred_body = pred[0]
        pred_face = pred[1]
        pred_hair = pred[2]
        pred_body = torch.nn.functional.interpolate(pred_body, (h,w), mode='bilinear', align_corners=False)
        pred_face = torch.nn.functional.interpolate(pred_face, (h,w), mode='bilinear', align_corners=False)
        pred_hair = torch.nn.functional.interpolate(pred_hair, (h,w), mode='bilinear', align_corners=False)
        pred_softmax = torch.softmax(pred_body, dim=1)
        #mask_soft, mask = torch.max(pred_softmax, dim=1)
        #mask_np = mask.squeeze(0).cpu().numpy().astype(np.uint8)

        pred_face_softmax = torch.softmax(pred_face, dim=1)
        
        pred_hair_softmax = torch.softmax(pred_hair, dim=1)
        #mask_face_soft, face_mask = torch.max(pred_face_softmax, dim=1)
        #face_np = face_mask.squeeze(0).cpu().numpy().astype(np.uint8)
        
        #photo_soft, photo = torch.max(pred_softmax, dim=2)
        #photo_np = photo.squeeze(0).cpu().numpy().astype(np.uint8)
        return pred_softmax.squeeze(0).cpu().numpy()[1], pred_face_softmax.squeeze(0).cpu().numpy()[1], pred_hair_softmax.squeeze(0).cpu().numpy()[1]
def addWeight(mask, image, label_value=1.0):
    shape = image.shape

    mask = cv2.resize(mask, (shape[1], shape[0]))
    if len(mask.shape) >= 3:
        mask = (mask.mean(axis=2)==int(label_value)).astype(np.uint8)
    else:
        mask = (mask==int(label_value)).astype(np.uint8)

    img_blue = np.zeros([shape[0], shape[1], 3], np.uint8)
    img_blue[:, :, 1] = np.zeros([shape[0], shape[1]]) + 255
    img_blue[mask == 0] = 0

    return cv2.addWeighted(img_blue, 0.6, image, 0.4, 0)

def get_crop(mask, img):
    mask[mask >= 127] = 255
    mask[mask < 127] = 0
    hei,wid = img.shape[:2]

    _, contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(contours[0])
    x = x - 10
    y = y - 10
    w = w + 10
    h = h + 10
    x = min(0, x)
    y = min(0, y)
    w = min(w, wid - x)
    h = min(h, hei - y)
    img_crop = img[y:y+h, x:x+w]
    return x,y,w,h,img_crop
    

def expand_dim(mask):
    mask = np.expand_dims(mask, 2)
    mask = mask.repeat(3, axis=2)
    return mask
if __name__ == '__main__':
    args = parser.parse_args()

    model_type = 'hrnet_ocr'
    weight_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ckpt', 'final_state.pth')
    input_size = (512, 512)
    model = EvalNet(model_type, weight_path, input_size=input_size)
    root_folder = args.input_dir
    out_folder = args.result_dir
    os.makedirs(out_folder, exist_ok=True)
    print("process %s" % root_folder)
    image_paths = sorted(glob.glob(os.path.join(root_folder, "*.*g")))
    print("lens of image:", len(image_paths))
    out = out_folder
    os.makedirs(out, exist_ok=True)
    assert (args.output_comp or args.output_mask)
    
    if args.output_comp:
        os.makedirs(os.path.join(out, 'comp'), exist_ok=True)
    
    if args.output_mask:
        os.makedirs(os.path.join(out, 'mask'), exist_ok=True)


    image_path = image_paths[0]
    img = cv2.imread(image_path)
    h, w, c = img.shape

    if args.output_comp:
            if args.background_img is not None:
                bg = cv2.imread(args.background_img)
                bg = cv2.resize(bg, (w, h))

    SegV = SegVideoPostproc2(h, w, 3)


    for i in tqdm(range(len(image_paths))):
        image_path = image_paths[i]
        img = cv2.imread(image_path)
        body, face, hair = model(img)
        h, w, c = img.shape

        
        if args.ridge_regression:
            body, show_img = SegV.ridge_regression(body, img)
        elif args.guide_filter:
            mask = SegV.dilate(body)
            mask = SegV.guided_filter(img, mask)

            out_mask = cv2.GaussianBlur(mask, (3, 3), 0)
            out_mask = cv2.resize(out_mask, (img.shape[1], img.shape[0]))
            out_mask[out_mask < 0.3] = 0.0
            out_mask = 2.5 * out_mask - 0.75
            out_mask[out_mask > 1.0] = 1.0 
            out_mask[out_mask < 0.0] = 0.0
            
            body = out_mask
            show_img = img
        else:
            show_img = img



        basename = os.path.splitext(os.path.basename(image_path))[0]

        if args.output_comp:
            if args.background_img is not None:
                pass
            else:
                bg = np.ones_like(img) * np.array(args.background_bgr)
            out_img = expand_dim(body) * show_img + (1 - expand_dim(body)) * bg
            cv2.imwrite(os.path.join(out, 'comp', basename + '.png'), out_img)

        if args.output_mask:
            if args.mask_mode == 'float':
                show_body = expand_dim(body) * 255 
            elif args.mask_mode == 'binary':
                show_body = body
                show_body[body>0.5] = 1
                show_body[body<=0.5] = 0
            elif args.mask_mode == 'binary-255':
                show_body = body
                show_body[body>0.5] = 1
                show_body[body<=0.5] = 0
                show_body *= 255
                
            cv2.imwrite(os.path.join(out, 'mask', basename + '.png'), show_body.round().astype(np.uint8))
