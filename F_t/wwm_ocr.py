

# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import subprocess

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
# sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../")))

os.environ["FLAGS_allocator_strategy"] = "auto_growth"
import cv2
import json
import numpy as np
import time
import logging
from copy import deepcopy

from paddle.utils import try_import
from paddleocr.ppocr.utils.utility import check_and_read
from paddleocr.ppocr.utils.logging import get_logger
from paddleocr.ppocr.utils.visual import draw_ser_results, draw_re_results
from paddleocr.tools.infer.predict_system import TextSystem
from paddleocr.tools.infer.predict_rec import TextRecognizer
from paddleocr.ppstructure.layout.predict_layout import LayoutPredictor
from paddleocr.ppstructure.table.predict_table import TableSystem, to_excel
from paddleocr.ppstructure.utility import parse_args, draw_structure_result, cal_ocr_word_box

logger = get_logger()


class StructureSystem(object):
    def __init__(self, args):
        self.mode = args.mode
        self.recovery = args.recovery

        self.image_orientation_predictor = None
        if args.image_orientation:
            import paddleclas
            self.image_orientation_predictor = paddleclas.PaddleClas(
                model_name="text_image_orientation"
            )

        if self.mode == "structure":
            if not args.show_log:
                logger.setLevel(logging.INFO)
            if args.layout == False and args.ocr == True:
                args.ocr = False
                logger.warning(
                    "When args.layout is false, args.ocr is automatically set to false"
                )
            # init model
            self.layout_predictor = None
            self.text_system = None
            self.table_system = None
            self.formula_system = None
            if args.layout:
                self.layout_predictor = LayoutPredictor(args)
                if args.ocr:
                    self.text_system = TextSystem(args)
            if args.table:
                if self.text_system is not None:
                    self.table_system = TableSystem(
                        args,
                        self.text_system.text_detector,
                        self.text_system.text_recognizer,
                    )
                else:
                    self.table_system = TableSystem(args)
            if args.formula:
                args_formula = deepcopy(args)
                args_formula.rec_algorithm = args.formula_algorithm
                args_formula.rec_model_dir = args.formula_model_dir
                args_formula.rec_char_dict_path = args.formula_char_dict_path
                args_formula.rec_batch_num = args.formula_batch_num
                self.formula_system = TextRecognizer(args_formula)

        elif self.mode == "kie":
            from ppstructure.kie.predict_kie_token_ser_re import SerRePredictor

            self.kie_predictor = SerRePredictor(args)

        self.return_word_box = args.return_word_box

    def __call__(self, img, return_ocr_result_in_table=False, img_idx=0):
        time_dict = {
            "image_orientation": 0,
            "layout": 0,
            "table": 0,
            "table_match": 0,
            "formula": 0,
            "det": 0,
            "rec": 0,
            "kie": 0,
            "all": 0,
        }
        start = time.time()

        if self.image_orientation_predictor is not None:
            tic = time.time()
            cls_result = self.image_orientation_predictor.predict(input_data=img)
            cls_res = next(cls_result)
            angle = cls_res[0]["label_names"][0]
            cv_rotate_code = {
                "90": cv2.ROTATE_90_COUNTERCLOCKWISE,
                "180": cv2.ROTATE_180,
                "270": cv2.ROTATE_90_CLOCKWISE,
            }
            if angle in cv_rotate_code:
                img = cv2.rotate(img, cv_rotate_code[angle])
            toc = time.time()
            time_dict["image_orientation"] = toc - tic

        if self.mode == "structure":
            ori_im = img.copy()
            if self.layout_predictor is not None:
                layout_res, elapse = self.layout_predictor(img)
                time_dict["layout"] += elapse
            else:
                h, w = ori_im.shape[:2]
                layout_res = [dict(bbox=None, label="table", score=0.0)]

            # As reported in issues such as #10270 and #11665, the old
            # implementation, which recognizes texts from the layout regions,
            # has problems with OCR recognition accuracy.
            #
            # To enhance the OCR recognition accuracy, we implement a patch fix
            # that first use text_system to detect and recognize all text information
            # and then filter out relevant texts according to the layout regions.
            text_res = None
            if self.text_system is not None:
                text_res, ocr_time_dict = self._predict_text(img)
                time_dict["det"] += ocr_time_dict["det"]
                time_dict["rec"] += ocr_time_dict["rec"]

            res_list = []
            for region in layout_res:
                res = ""
                if region["bbox"] is not None:
                    x1, y1, x2, y2 = region["bbox"]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    roi_img = ori_im[y1:y2, x1:x2, :]
                else:
                    x1, y1, x2, y2 = 0, 0, w, h
                    roi_img = ori_im
                bbox = [x1, y1, x2, y2]

                if region["label"] == "table":
                    if self.table_system is not None:
                        res, table_time_dict = self.table_system(
                            roi_img, return_ocr_result_in_table
                        )
                        time_dict["table"] += table_time_dict["table"]
                        time_dict["table_match"] += table_time_dict["match"]
                        time_dict["det"] += table_time_dict["det"]
                        time_dict["rec"] += table_time_dict["rec"]

                elif region["label"] == "equation" and self.formula_system is not None:
                    latex_res, formula_time = self.formula_system([roi_img])
                    time_dict["formula"] += formula_time
                    res = {"latex": latex_res[0]}

                else:
                    if text_res is not None:
                        # Filter the text results whose regions intersect with the current layout bbox.
                        res = self._filter_text_res(text_res, bbox)

                res_list.append(
                    {
                        "type": region["label"].lower(),
                        "bbox": bbox,
                        "img": roi_img,
                        "res": res,
                        "img_idx": img_idx,
                        "score": region["score"],
                    }
                )

            end = time.time()
            time_dict["all"] = end - start
            return res_list, time_dict

        elif self.mode == "kie":
            re_res, elapse = self.kie_predictor(img)
            time_dict["kie"] = elapse
            time_dict["all"] = elapse
            return re_res[0], time_dict

        return None, None

    def _predict_text(self, img):
        filter_boxes, filter_rec_res, ocr_time_dict = self.text_system(img)

        # remove style char,
        # when using the recognition model trained on the PubtabNet dataset,
        # it will recognize the text format in the table, such as <b>
        style_token = [
            "<strike>",
            "<strike>",
            "<sup>",
            "</sub>",
            "<b>",
            "</b>",
            "<sub>",
            "</sup>",
            "<overline>",
            "</overline>",
            "<underline>",
            "</underline>",
            "<i>",
            "</i>",
        ]
        res = []
        for box, rec_res in zip(filter_boxes, filter_rec_res):
            rec_str, rec_conf = rec_res[0], rec_res[1]
            for token in style_token:
                if token in rec_str:
                    rec_str = rec_str.replace(token, "")
            if self.return_word_box:
                word_box_content_list, word_box_list = cal_ocr_word_box(
                    rec_str, box, rec_res[2]
                )
                res.append(
                    {
                        "text": rec_str,
                        "confidence": float(rec_conf),
                        "text_region": box.tolist(),
                        "text_word": word_box_content_list,
                        "text_word_region": word_box_list,
                    }
                )
            else:
                res.append(
                    {
                        "text": rec_str,
                        "confidence": float(rec_conf),
                        "text_region": box.tolist(),
                    }
                )
        return res, ocr_time_dict

    def _filter_text_res(self, text_res, bbox):
        res = []
        for r in text_res:
            box = r["text_region"]
            rect = box[0][0], box[0][1], box[2][0], box[2][1]
            if self._has_intersection(bbox, rect):
                res.append(r)
        return res

    def _has_intersection(self, rect1, rect2):
        x_min1, y_min1, x_max1, y_max1 = rect1
        x_min2, y_min2, x_max2, y_max2 = rect2
        if x_min1 > x_max2 or x_max1 < x_min2:
            return False
        if y_min1 > y_max2 or y_max1 < y_min2:
            return False
        return True




class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def get_image_file_list(path):
    img_ends = ['png', 'jpg']
    if path.split('.')[-1] in img_ends:
        return [path]
    elif os.path.isdir(path):
        names = [i for i in os.listdir(path) if i.split('.')[-1] in img_ends]
        return [os.path.join(path, name) for name in names]
    else:
        print(f'args.image_dir is wrong : {path}')



class core:
    def __init__(self, args):
        self.args = self.add_default_args(args)
        self.structure_sys = StructureSystem(self.args)

    def add_default_args(self, args):
        args.det_model_dir = "./ppstructure/inference/en_PP-OCRv3_det_infer"
        args.rec_model_dir = "./ppstructure/inference/en_PP-OCRv3_rec_infer"
        args.rec_char_dict_path = "./ppocr/utils/en_dict.txt"
        args.table_model_dir = "./ppstructure/inference/en_ppstructure_mobile_v2.0_SLANet_infer"
        args.table_char_dict_path = './ppocr/utils/dict/table_structure_dict.txt'
        args.layout_model_dir = './ppstructure/inference/picodet_lcnet_x1_0_fgd_layout_infer'
        args.layout_dict_path='./ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt'
        args.vis_font_path = './doc/fonts/simfang.ttf'
        args.ocr = True
        args.return_word_box=True
        args.image_orientation=True
        args.show_log = False
        args.use_gpu = False
        return args


    def deal_one_image(self, image_file_or_imgbgr, save_folder:str=None, needprint:bool=True):
        if isinstance(image_file_or_imgbgr, str):
            img = cv2.imread(image_file_or_imgbgr)
        else:
            img = image_file_or_imgbgr
        res, time_dict = self.structure_sys(img, img_idx=0)
        res = res[0]['res']
        
        if save_folder is not None and isinstance(image_file_or_imgbgr, str):
            os.makedirs(save_folder, exist_ok=True)
            savepath = os.path.join(save_folder, os.path.basename(image_file_or_imgbgr)[:-4] +'.json')
            with open(savepath, 'w') as json_file:
                json.dump(res, json_file, cls=NumpyEncoder, ensure_ascii=False, indent=4)
            if needprint:
                print(f'Saved to {savepath}')

        return res


def get_args():
    args = parse_args()
    return args
