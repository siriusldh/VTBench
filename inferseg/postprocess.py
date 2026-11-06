import numpy as np
import cv2
from numpy.linalg import inv


class SegVideoPostproc(object):
    """docstring for SegVideoPostproc"""
    def __init__(self):
        super(SegVideoPostproc, self).__init__()
        self.mask_list = []
        self.mask_list_small = []

    def reset(self):
        self.mask_list = []
        self.mask_list_small = []

    def dilate(self, logit):
        mask = (logit * 255).astype(np.uint8)
        mask[mask >= 127] = 255
        mask[mask < 127] = 0
        mask = self.removeoutlier(mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (3, 3))
        dilation =  cv2.dilate(mask, kernel, iterations = 1)
        dilation = dilation / 255.0
        logit = logit * dilation
        return logit

    def removeoutlier(self, mask):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        n = len(contours)
        if n > 0:
            areas = []
            for i in range(n):
                areas.append(cv2.contourArea(contours[i]))
            max_area = max(areas)
            outlier_areas = []
            for i in range(n):
                if areas[i] < 0.1 * max_area:
                    outlier_areas.append(contours[i])
            cv2.fillPoly(mask, outlier_areas, (0, 0, 0))
        return mask



    def guided_filter(self, img, mask, r=5, eps=1e-6):

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
        gray_img_small = cv2.resize(gray_img, (mask.shape[1], mask.shape[0]))

        blur_I1 = cv2.blur(gray_img_small, (r, r))
        blur_I2 = cv2.blur(gray_img_small * gray_img_small, (r, r))
        var_I = blur_I2 - blur_I1 * blur_I1


        blur_mask1 = cv2.blur(mask, (r, r))
        blur_Im = cv2.blur(gray_img_small * mask, (r, r))
        cov_Im = blur_Im - blur_I1 * blur_mask1

        a = cov_Im / (var_I + eps)
        b = blur_mask1 - a * blur_I1

        a = cv2.blur(a, (r, r))
        b = cv2.blur(b, (r, r))

        # mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        # a = cv2.resize(a, (img.shape[1], img.shape[0]))
        # b = cv2.resize(b, (img.shape[1], img.shape[0]))
        q = a * gray_img_small + b
        mask = mask * q
        return mask

    def ridge_regression(self, logit, img):
        # logit[logit >= 0.5] = 1.0
        # logit[logit < 0.5] = 0.0
        # cv2.imwrite("test_videos/logit.png", (logit * 255).astype(np.uint8))
        mask = self.dilate(logit)
        # cv2.imwrite("test_videos/dilate_mask.png", (mask * 255).astype(np.uint8))
        mask = self.guided_filter(img, mask)

        # cv2.imwrite("test_videos/guided_filter_mask.png", (mask * 255).astype(np.uint8))
        mask_small = cv2.resize(mask, (16, 16))
        # print("mask_small:", mask_small)
        mask_small_vec = mask_small.reshape(256, 1)
        # print("mask_small_vec:", mask_small_vec)

        self.mask_list.append(mask)
        self.mask_list_small.append(mask_small_vec)
        m = len(self.mask_list_small)
        if m > 4:
            self.mask_list_small = self.mask_list_small[-4:]
            self.mask_list = self.mask_list[-4:]
        n = len(self.mask_list_small)
        
        At = np.matrix(np.array(self.mask_list_small).reshape(n, 256))
        # print("At:", At[-1,:])
        lamb = 1.0
        mask_small_cv = mask_small_vec.copy()
        
        I = np.matrix(np.eye(n))
        XTX = At * At.T
        # a = np.linalg.pinv(XTX + lamb * I)
        # b = a * At.T
        # c = b * mask_small_cv
        # print(XTX.shape)
        # print(I.shape)
        # print(mask_small_cv.shape)
        alpha = (XTX + lamb * I).I * At * mask_small_cv 
        # alpha = (At * A + lamb * np.eye(n).astype(np.float32)) * At * mask_small_cv
        out_mask = np.zeros((mask.shape[0], mask.shape[1])).astype(np.float32)
        # print(alpha)
        for i in range(n):
            out_mask += alpha[i, 0] * self.mask_list[i]

        out_mask = cv2.GaussianBlur(out_mask, (3, 3), 0)
        out_mask = cv2.resize(out_mask, (img.shape[1], img.shape[0]))
        # out_mask[out_mask >= 0.5] = 1.0
        # out_mask[out_mask < 0.5] = 0.0
        out_mask[out_mask < 0.3] = 0.0
        out_mask = 2.5 * out_mask - 0.75
        out_mask[out_mask > 1.0] = 1.0 
        out_mask[out_mask < 0.0] = 0.0
        return out_mask

class SegVideoPostprocLocal(object):
    """docstring for SegVideoPostproc"""
    def __init__(self):
        super(SegVideoPostprocLocal, self).__init__()
        self.mask_list = []
        self.mask_list_1 = []
        self.mask_list_2 = []
        self.mask_list_3 = []
        self.mask_list_4 = []
        self.mask_list_small_1 = []
        self.mask_list_small_2 = []
        self.mask_list_small_3 = []
        self.mask_list_small_4 = []

    def reset(self):
        self.mask_list = []
        self.mask_list_1 = []
        self.mask_list_2 = []
        self.mask_list_3 = []
        self.mask_list_4 = []
        self.mask_list_small_1 = []
        self.mask_list_small_2 = []
        self.mask_list_small_3 = []
        self.mask_list_small_4 = []

    def dilate(self, logit):
        mask = (logit * 255).astype(np.uint8)
        mask[mask >= 127] = 255
        mask[mask < 127] = 0
        mask = self.removeoutlier(mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (3, 3))
        dilation =  cv2.dilate(mask, kernel, iterations = 1)
        dilation = dilation / 255.0
        logit = logit * dilation
        return logit

    def rg_alpha(self, mask_list_small, mask_small_vec, mask_list):
        n = len(mask_list_small)
        
        At = np.matrix(np.array(mask_list_small).reshape(n, 720))
        # print("At:", At[-1,:])
        lamb = 1.0
        mask_small_cv = mask_small_vec.copy()
        
        I = np.matrix(np.eye(n))
        XTX = At * At.T
        alpha = (XTX + lamb * I).I * At * mask_small_cv
        out_mask = np.zeros((mask_list[0].shape[0], mask_list[0].shape[1])).astype(np.float32)
        # print(alpha)
        for i in range(n):
            out_mask += alpha[i, 0] * mask_list[i]
        return out_mask



    def removeoutlier(self, mask):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        n = len(contours)
        if n > 0:
            areas = []
            for i in range(n):
                areas.append(cv2.contourArea(contours[i]))
            max_area = max(areas)
            outlier_areas = []
            for i in range(n):
                if areas[i] < 0.1 * max_area:
                    outlier_areas.append(contours[i])
            cv2.fillPoly(mask, outlier_areas, (0, 0, 0))
        return mask



    def guided_filter(self, img, mask, r=5, eps=1e-6):

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
        gray_img_small = cv2.resize(gray_img, (mask.shape[1], mask.shape[0]))

        blur_I1 = cv2.blur(gray_img_small, (r, r))
        blur_I2 = cv2.blur(gray_img_small * gray_img_small, (r, r))
        var_I = blur_I2 - blur_I1 * blur_I1


        blur_mask1 = cv2.blur(mask, (r, r))
        blur_Im = cv2.blur(gray_img_small * mask, (r, r))
        cov_Im = blur_Im - blur_I1 * blur_mask1

        a = cov_Im / (var_I + eps)
        b = blur_mask1 - a * blur_I1

        a = cv2.blur(a, (r, r))
        b = cv2.blur(b, (r, r))

        # mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        # a = cv2.resize(a, (img.shape[1], img.shape[0]))
        # b = cv2.resize(b, (img.shape[1], img.shape[0]))
        q = a * gray_img_small + b
        mask = mask * q
        return mask

    def process(self, logit, img):
        mask = self.dilate(logit)
        mask = self.guided_filter(img, mask)
        mask_small = cv2.resize(mask, (144, 80))
        mask_small_1 = mask_small[:72, :40]
        mask_small_vec_1 = cv2.resize(mask_small_1, (16, 16))
        mask_small_vec_1 = mask_small_vec_1.reshape(256, 1)
        
        mask_small_2 = mask_small[:72, 40:]
        mask_small_vec_2 = cv2.resize(mask_small_2, (16, 16))
        mask_small_vec_2 = mask_small_vec_2.reshape(256, 1)
        
        mask_small_3 = mask_small[72:, :40]
        mask_small_vec_3 = cv2.resize(mask_small_3, (16, 16))
        mask_small_vec_3 = mask_small_vec_3.reshape(256, 1)
        
        mask_small_4 = mask_small[72:, 40:]
        mask_small_vec_4 = cv2.resize(mask_small_4, (16, 16))
        mask_small_vec_4 = mask_small_vec_4.reshape(256, 1)

        self.mask_list_small_1.append(mask_small_vec_1)
        self.mask_list_small_2.append(mask_small_vec_2)
        self.mask_list_small_3.append(mask_small_vec_3)
        self.mask_list_small_4.append(mask_small_vec_4)
        self.mask_list_1.append(mask_small_1)
        self.mask_list_2.append(mask_small_2)
        self.mask_list_3.append(mask_small_3)
        self.mask_list_4.append(mask_small_4)
        self.mask_list.append(mask)
        if m > 4:
            self.mask_list_small_1 = self.mask_list_small[-4:]
            self.mask_list_small_2 = self.mask_list_small[-4:]
            self.mask_list_small_3 = self.mask_list_small[-4:]
            self.mask_list_small_4 = self.mask_list_small[-4:]
            self.mask_list = self.mask_list[-4:]
            self.mask_list_1 = self.mask_list_1[-4:]
            self.mask_list_2 = self.mask_list_2[-4:]
            self.mask_list_3 = self.mask_list_3[-4:]
            self.mask_list_4 = self.mask_list_4[-4:]

        out_mask1 = self.rg_alpha(self.mask_list_small_1, mask_small_vec_1, mask_list_1)
        out_mask2 = self.rg_alpha(self.mask_list_small_2, mask_small_vec_2, mask_list_2)

        out_mask3 = self.rg_alpha(self.mask_list_small_3, mask_small_vec_3, mask_list_3)
        out_mask4 = self.rg_alpha(self.mask_list_small_4, mask_small_vec_4, mask_list_4)

        out_mask = np.zeros((80, 144)).astype(np.float32)
        out_mask[:72, :40] = out_mask1
        out_mask[:72, 40:] = out_mask2
        out_mask[72:, :40] = out_mask3
        out_mask[72:, 40:] = out_mask4

        out_mask = cv2.GaussianBlur(out_mask, (3, 3), 0)
        out_mask = cv2.resize(out_mask, (img.shape[1], img.shape[0]))
        # out_mask[out_mask >= 0.5] = 1.0
        # out_mask[out_mask < 0.5] = 0.0
#         out_mask[out_mask < 0.3] = 0.0
#         out_mask = 2.5 * out_mask - 0.75
#         out_mask[out_mask > 1.0] = 1.0 
#         out_mask[out_mask < 0.0] = 0.0
        return out_mask






    def ridge_regression(self, logit, img):
        # logit[logit >= 0.5] = 1.0
        # logit[logit < 0.5] = 0.0
        # cv2.imwrite("test_videos/logit.png", (logit * 255).astype(np.uint8))
        mask = self.dilate(logit)
        # cv2.imwrite("test_videos/dilate_mask.png", (mask * 255).astype(np.uint8))
        mask = self.guided_filter(img, mask)

        # cv2.imwrite("test_videos/guided_filter_mask.png", (mask * 255).astype(np.uint8))
        mask_small = cv2.resize(mask, (16, 16))
        # print("mask_small:", mask_small)
        mask_small_vec = mask_small.reshape(256, 1)
        # print("mask_small_vec:", mask_small_vec)

        self.mask_list.append(mask)
        self.mask_list_small.append(mask_small_vec)
        m = len(self.mask_list_small)
        if m > 4:
            self.mask_list_small = self.mask_list_small[-4:]
            self.mask_list = self.mask_list[-4:]
        n = len(self.mask_list_small)
        
        At = np.matrix(np.array(self.mask_list_small).reshape(n, 256))
        # print("At:", At[-1,:])
        lamb = 1.0
        mask_small_cv = mask_small_vec.copy()
        
        I = np.matrix(np.eye(n))
        XTX = At * At.T
        # a = np.linalg.pinv(XTX + lamb * I)
        # b = a * At.T
        # c = b * mask_small_cv
        # print(XTX.shape)
        # print(I.shape)
        # print(mask_small_cv.shape)
        alpha = (XTX + lamb * I).I * At * mask_small_cv 
        # alpha = (At * A + lamb * np.eye(n).astype(np.float32)) * At * mask_small_cv
        out_mask = np.zeros((mask.shape[0], mask.shape[1])).astype(np.float32)
        # print(alpha)
        for i in range(n):
            out_mask += alpha[i, 0] * self.mask_list[i]

        out_mask = cv2.GaussianBlur(out_mask, (3, 3), 0)
        out_mask = cv2.resize(out_mask, (img.shape[1], img.shape[0]))
        # out_mask[out_mask >= 0.5] = 1.0
        # out_mask[out_mask < 0.5] = 0.0
        out_mask[out_mask < 0.3] = 0.0
        out_mask = 2.5 * out_mask - 0.75
        out_mask[out_mask > 1.0] = 1.0 
        out_mask[out_mask < 0.0] = 0.0
        return out_mask

class SegVideoPostproc2(object):
    """docstring for SegVideoPostproc"""
    def __init__(self, h, w, nums):
        super(SegVideoPostproc2, self).__init__()
        self.h = h
        self.w = w
        self._blured = np.zeros((h , w))
        self._r = np.zeros((h, w))
        self._k = np.zeros((h, w))
        self._correction = np.zeros((h, w))
        self._predicated = np.zeros((h, w))
        self.mask_list = []
        self.mask_list_small = []
        self.img_list = []
        self.nums = nums

    def reset(self):
        self._blured = np.zeros((h , w))
        self._r = np.zeros((h, w))
        self._k = np.zeros((h, w))
        self._correction = np.zeros((h, w))
        self._predicated = np.zeros((h, w))
        self.mask_list = []
        self.mask_list_small = []

    def dilate(self, logit):
        mask = (logit * 255).astype(np.uint8)
        mask[mask >= 127] = 255
        mask[mask < 127] = 0
        mask = self.removeoutlier(mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (3, 3))
        dilation =  cv2.dilate(mask, kernel, iterations = 1)
        dilation = dilation / 255.0
        logit = logit * dilation
        return logit

    def removeoutlier(self, mask):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        n = len(contours)
        if n > 0:
            areas = []
            for i in range(n):
                areas.append(cv2.contourArea(contours[i]))
            max_area = max(areas)
            outlier_areas = []
            for i in range(n):
                if areas[i] < 0.1 * max_area:
                    outlier_areas.append(contours[i])
            cv2.fillPoly(mask, outlier_areas, (0, 0, 0))
        return mask



    def guided_filter(self, img, mask, r=5, eps=1e-6):

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
        gray_img_small = cv2.resize(gray_img, (mask.shape[1], mask.shape[0]))

        blur_I1 = cv2.blur(gray_img_small, (r, r))
        blur_I2 = cv2.blur(gray_img_small * gray_img_small, (r, r))
        var_I = blur_I2 - blur_I1 * blur_I1


        blur_mask1 = cv2.blur(mask, (r, r))
        blur_Im = cv2.blur(gray_img_small * mask, (r, r))
        cov_Im = blur_Im - blur_I1 * blur_mask1

        a = cov_Im / (var_I + eps)
        b = blur_mask1 - a * blur_I1

        a = cv2.blur(a, (r, r))
        b = cv2.blur(b, (r, r))

        # mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        # a = cv2.resize(a, (img.shape[1], img.shape[0]))
        # b = cv2.resize(b, (img.shape[1], img.shape[0]))
        q = a * gray_img_small + b
        mask = mask * q
        return mask

    def kalman_regression(self, logit, img):
        mask = self.dilate(logit)
        mask = self.guided_filter(img, mask)
        out_mask = np.zeros((mask.shape[0], mask.shape[1])).astype(np.float32)
        # for i in range(self.h):
        #     blur_curr = 0.0
        #     delta = 0.0
        #     pred = 0.0
        #     r = 0.0
        #     k = 0.0
        #     predicated = 0
        #     for j in range(self.w):
        #         blur_curr = mask[i][j]
        #         delta = self._blured[i][j] - blur_curr
        #         self._blured[i][j] = blur_curr
        #         r = 1 + self._r[i][j] / (1 + self._k[i][j])
        #         self._r[i][j] = r
        #         pred = self._correction[i][j] + 0.15 * delta ** 2
        #         k = pred / (pred + r)
        #         predicated = self._predicated[i][j] + k * (mask[i][j] - self._predicated[i][j])
        #         self._predicated[i][j] = predicated
        #         self._correction[i][j] = (1 - k) * pred
        #         out_mask[i][j] = predicated
        delta = (self._blured - mask) * 255
        self._r = 1 + self._r / ( 1 + self._k)
        pred = self._correction + 0.1 * delta ** 2
        k = pred / (pred + self._r)
        self._k = k
        predicated = self._predicated + k * (mask - self._predicated)
        self._predicated = predicated
        self._correction = (1 - k) * pred

        out_mask = predicated

        out_mask = cv2.GaussianBlur(out_mask, (3, 3), 1.5)
        out_mask = cv2.resize(out_mask, (img.shape[1], img.shape[0]))

        return out_mask


    def ridge_regression(self, logit, img):
        # logit[logit >= 0.5] = 1.0
        # logit[logit < 0.5] = 0.0
        # cv2.imwrite("test_videos/logit.png", (logit * 255).astype(np.uint8))
        mask = self.dilate(logit)
        # cv2.imwrite("test_videos/dilate_mask.png", (mask * 255).astype(np.uint8))
        mask = self.guided_filter(img, mask)

        # cv2.imwrite("test_videos/guided_filter_mask.png", (mask * 255).astype(np.uint8))
        mask_small = cv2.resize(mask, (16, 16))
        # print("mask_small:", mask_small)
        mask_small_vec = mask_small.reshape(256, 1)
        # print("mask_small_vec:", mask_small_vec)

        self.mask_list.append(mask)
        self.mask_list_small.append(mask_small_vec)
        self.img_list.append(img)
        m = len(self.mask_list_small)
        nums = self.nums
        if m <=nums:
            out_mask = mask
            out_img = img
        else:
            self.mask_list_small = self.mask_list_small[-nums:]
            self.mask_list = self.mask_list[-nums:]
            self.img_list = self.img_list[-nums:]
            n = len(self.mask_list_small)
            
            At = np.matrix(np.array(self.mask_list_small).reshape(n, 256))
            # print("At:", At[-1,:])
            lamb = 1.0
            mask_small_cv = self.mask_list_small[int(nums / 2)].copy()
            
            I = np.matrix(np.eye(n))
            XTX = At * At.T
            # a = np.linalg.pinv(XTX + lamb * I)
            # b = a * At.T
            # c = b * mask_small_cv
            # print(XTX.shape)
            # print(I.shape)
            # print(mask_small_cv.shape)
            alpha = (XTX + lamb * I).I * At * mask_small_cv 
            # alpha = (At * A + lamb * np.eye(n).astype(np.float32)) * At * mask_small_cv
            out_mask = np.zeros((mask.shape[0], mask.shape[1])).astype(np.float32)
            out_img = self.img_list[int(nums / 2)]
            # print(alpha)
            for i in range(n):
                out_mask += alpha[i, 0] * self.mask_list[i]

        out_mask = cv2.GaussianBlur(out_mask, (3, 3), 0)
        out_mask = cv2.resize(out_mask, (img.shape[1], img.shape[0]))
        # out_mask[out_mask >= 0.5] = 1.0
        # out_mask[out_mask < 0.5] = 0.0
        out_mask[out_mask < 0.3] = 0.0
        out_mask = 2.5 * out_mask - 0.75
        out_mask[out_mask > 1.0] = 1.0 
        out_mask[out_mask < 0.0] = 0.0
        return out_mask, out_img

import math

class LowPassFilter(object):

    def __init__(self, alpha):
        self.__setAlpha(alpha)
        self.__y = self.__s = None

    def __setAlpha(self, alpha):
        # alpha = float(alpha)
        # if alpha<=0 or alpha>1.0:
        #     raise ValueError("alpha (%s) should be in (0.0, 1.0]"%alpha)
        self.__alpha = alpha

    def __call__(self, value, timestamp=None, alpha=None):        
        if alpha is not None:
            self.__setAlpha(alpha)
        if self.__y is None:
            s = value
        else:
            s = self.__alpha*value + (1.0-self.__alpha)*self.__s
        self.__y = value
        self.__s = s
        return s

    def lastValue(self):
        return self.__y


class SegVideoPostproc3(object):
    """docstring for SegVideoPostproc"""
    def __init__(self, h, w, freq=25, mincutoff=1.0, beta=1.0, dcutoff=1.0):
        super(SegVideoPostproc3, self).__init__()
        self.h = h
        self.w = w
        self.__freq = float(freq)
        self.__mincutoff = np.ones((h, w)) * mincutoff
        self.__beta = float(beta)
        self.__dcutoff = np.ones((h, w)) * dcutoff
        self.__x = LowPassFilter(self.__alpha(self.__mincutoff))
        self.__dx = LowPassFilter(self.__alpha(self.__dcutoff))
        self.__lasttime = None
        self.disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self.disflow.setFinestScale(3)
        self.pre_gray = None

    def cal_flow(self, img, mask):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img_small = cv2.resize(gray_img, (mask.shape[1], mask.shape[0]))
        flow = np.zeros((mask.shape[0], mask.shape[1]))
        if self.pre_gray is None:
            self.pre_gray = gray_img_small
            return flow
        else:
            flow_fw = self.disflow.calc(self.pre_gray, gray_img_small, None)
            flow = flow_fw[:, :, 1] **2 + flow_fw[:, :, 0] ** 2
            flow = np.sqrt(flow)
            self.pre_gray = gray_img_small
            return flow


    def __alpha(self, cutoff):
        te = 1.0 / self.__freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te) 
    
    def reset(self):
        self.__freq = float(120)
        self.__mincutoff = np.ones((h, w)) * 1.0
        self.__beta = float(1)
        self.__dcutoff = np.ones((h, w)) * 1.0
        self.__x = LowPassFilter(self.__alpha(self.__mincutoff))
        self.__dx = LowPassFilter(self.__alpha(self.__dcutoff))
        self.__lasttime = None
        self.pre_gray = None

    def dilate(self, logit):
        mask = (logit * 255).astype(np.uint8)
        mask[mask >= 127] = 255
        mask[mask < 127] = 0
        mask = self.removeoutlier(mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (3, 3))
        dilation =  cv2.dilate(mask, kernel, iterations = 1)
        dilation = dilation / 255.0
        logit = logit * dilation
        return logit

    def removeoutlier(self, mask):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        n = len(contours)
        if n > 0:
            areas = []
            for i in range(n):
                areas.append(cv2.contourArea(contours[i]))
            max_area = max(areas)
            outlier_areas = []
            for i in range(n):
                if areas[i] < 0.1 * max_area:
                    outlier_areas.append(contours[i])
            cv2.fillPoly(mask, outlier_areas, (0, 0, 0))
        return mask


    def guided_filter(self, img, mask, r=5, eps=1e-6):

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
        gray_img_small = cv2.resize(gray_img, (mask.shape[1], mask.shape[0]))


        blur_I1 = cv2.blur(gray_img_small, (r, r))
        blur_I2 = cv2.blur(gray_img_small * gray_img_small, (r, r))
        var_I = blur_I2 - blur_I1 * blur_I1


        blur_mask1 = cv2.blur(mask, (r, r))
        blur_Im = cv2.blur(gray_img_small * mask, (r, r))
        cov_Im = blur_Im - blur_I1 * blur_mask1

        a = cov_Im / (var_I + eps)
        b = blur_mask1 - a * blur_I1

        a = cv2.blur(a, (r, r))
        b = cv2.blur(b, (r, r))

        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        a = cv2.resize(a, (img.shape[1], img.shape[0]))
        b = cv2.resize(b, (img.shape[1], img.shape[0]))
        q = a * gray_img + b
        mask = mask * q
        return mask

    def OneEuroFilter(self, logit, img, timestamp=None):
        mask = self.dilate(logit)
        mask = self.guided_filter(img, mask)
        out_mask = np.zeros((mask.shape[0], mask.shape[1])).astype(np.float32)
        # flow = self.cal_flow(img, mask)
        if self.__lasttime and timestamp:
            self.__freq = 1.0 / (timestamp - self.__lasttime)
        self.__lasttime = timestamp
        
        #prev_mask = self.__x.lastValue()
        #dx = np.zeros((self.h, self.w)) if prev_mask is None else (mask - prev_mask) * self.__freq
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.resize(gray_img, (mask.shape[1], mask.shape[0]))
        if self.pre_gray is None:
            self.pre_gray = gray_img
            flow = np.zeros((gray_img.shape[0], gray_img.shape[1]), dtype=np.float32)
        else:
            flow_fw = self.disflow.calc(self.pre_gray, gray_img, None)
            flows = cv2.split(flow_fw)
            flow = cv2.magnitude(flows[0] / gray_img.shape[1], flows[1] / gray_img.shape[0])
            self.pre_gray = gray_img
        #print("flow:", (np.min(flow), np.max(flow)))
        #print("dx:", (np.min(dx), np.max(dx)))
        #dx = flow * self.__freq * 0.1 + dx * self.__freq * 0.9
        dx = flow * self.__freq
        edx = self.__dx(dx, timestamp, alpha=self.__alpha(self.__dcutoff))
        cutoff = self.__mincutoff + self.__beta * np.fabs(edx)

        out_mask = self.__x(mask, timestamp, alpha=self.__alpha(cutoff))
        #print("alpha:", (np.min(self.__alpha(cutoff)), np.max(self.__alpha(cutoff))))
        out_mask = cv2.GaussianBlur(out_mask, (3, 3), 1.5)
        out_mask = cv2.resize(out_mask, (img.shape[1], img.shape[0]))

        return out_mask, self.__alpha(cutoff)

    def kalman_regression(self, logit, img):
        mask = self.dilate(logit)
        mask = self.guided_filter(img, mask)
        out_mask = np.zeros((mask.shape[0], mask.shape[1])).astype(np.float32)
        delta = (self._blured - mask) * 255
        self._r = 1 + self._r / ( 1 + self._k)
        pred = self._correction + 0.1 * delta ** 2
        k = pred / (pred + self._r)
        self._k = k
        predicated = self._predicated + k * (mask - self._predicated)
        self._predicated = predicated
        self._correction = (1 - k) * pred

        out_mask = predicated

        out_mask = cv2.GaussianBlur(out_mask, (3, 3), 1.5)
        out_mask = cv2.resize(out_mask, (img.shape[1], img.shape[0]))

        return out_mask


    def ridge_regression(self, logit, img):
        # logit[logit >= 0.5] = 1.0
        # logit[logit < 0.5] = 0.0
        # cv2.imwrite("test_videos/logit.png", (logit * 255).astype(np.uint8))
        mask = self.dilate(logit)
        # cv2.imwrite("test_videos/dilate_mask.png", (mask * 255).astype(np.uint8))
        mask = self.guided_filter(img, mask)

        # cv2.imwrite("test_videos/guided_filter_mask.png", (mask * 255).astype(np.uint8))
        mask_small = cv2.resize(mask, (16, 16))
        # print("mask_small:", mask_small)
        mask_small_vec = mask_small.reshape(256, 1)
        # print("mask_small_vec:", mask_small_vec)

        self.mask_list.append(mask)
        self.mask_list_small.append(mask_small_vec)
        m = len(self.mask_list_small)
        if m > 4:
            self.mask_list_small = self.mask_list_small[-4:]
            self.mask_list = self.mask_list[-4:]
        n = len(self.mask_list_small)
        
        At = np.matrix(np.array(self.mask_list_small).reshape(n, 256))
        # print("At:", At[-1,:])
        lamb = 1.0
        mask_small_cv = mask_small_vec.copy()
        
        I = np.matrix(np.eye(n))
        XTX = At * At.T
        # a = np.linalg.pinv(XTX + lamb * I)
        # b = a * At.T
        # c = b * mask_small_cv
        # print(XTX.shape)
        # print(I.shape)
        # print(mask_small_cv.shape)
        alpha = (XTX + lamb * I).I * At * mask_small_cv 
        # alpha = (At * A + lamb  * np.eye(n).astype(np.float32)) * At * mask_small_cv
        out_mask = np.zeros((mask.shape[0], mask.shape[1])).astype(np.float32)
        # print(alpha)
        for i in range(n):
            out_mask += alpha[i, 0] * self.mask_list[i]

        out_mask = cv2.GaussianBlur(out_mask, (3, 3), 1.5)
        out_mask = cv2.resize(out_mask, (img.shape[1], img.shape[0]))
        # out_mask[out_mask >= 0.5] = 1.0
        # out_mask[out_mask < 0.5] = 0.0
        return out_mask


class SegVideoPostproc4(object):
    """docstring for SegVideoPostproc"""
    def __init__(self):
        super(SegVideoPostproc4, self).__init__()
        self.filter = seg_mask_filter()


    def reset(self):
        self.filter.reset()

    def dilate(self, logit):
        mask = (logit * 255).astype(np.uint8)
        mask[mask >= 127] = 255
        mask[mask < 127] = 0
        mask = self.removeoutlier(mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (3, 3))
        dilation =  cv2.dilate(mask, kernel, iterations = 1)
        dilation = dilation / 255.0
        logit = logit * dilation
        return logit

    def removeoutlier(self, mask):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        n = len(contours)
        if n > 0:
            areas = []
            for i in range(n):
                areas.append(cv2.contourArea(contours[i]))
            max_area = max(areas)
            outlier_areas = []
            for i in range(n):
                if areas[i] < 0.1 * max_area:
                    outlier_areas.append(contours[i])
            cv2.fillPoly(mask, outlier_areas, (0, 0, 0))
        return mask


    def guided_filter(self, img, mask, r=5, eps=1e-6):

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
        gray_img_small = cv2.resize(gray_img, (mask.shape[1], mask.shape[0]))


        blur_I1 = cv2.blur(gray_img_small, (r, r))
        blur_I2 = cv2.blur(gray_img_small * gray_img_small, (r, r))
        var_I = blur_I2 - blur_I1 * blur_I1


        blur_mask1 = cv2.blur(mask, (r, r))
        blur_Im = cv2.blur(gray_img_small * mask, (r, r))
        cov_Im = blur_Im - blur_I1 * blur_mask1

        a = cov_Im / (var_I + eps)
        b = blur_mask1 - a * blur_I1

        a = cv2.blur(a, (r, r))
        b = cv2.blur(b, (r, r))

        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        a = cv2.resize(a, (img.shape[1], img.shape[0]))
        b = cv2.resize(b, (img.shape[1], img.shape[0]))
        q = a * gray_img + b
        mask = mask * q
        return mask

    def seg_regression(self, logit, img):
        #mask = self.dilate(logit)
        # cv2.imwrite("test_videos/dilate_mask.png", (mask * 255).astype(np.uint8))
        #mask = self.guided_filter(img, mask)
        mask = self.filter(logit)
        #out_mask = cv2.GaussianBlur(mask, (3, 3), 1.5)
        out_mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        return out_mask


    def ridge_regression(self, logit, img):
        # logit[logit >= 0.5] = 1.0
        # logit[logit < 0.5] = 0.0
        # cv2.imwrite("test_videos/logit.png", (logit * 255).astype(np.uint8))
        mask = self.dilate(logit)
        # cv2.imwrite("test_videos/dilate_mask.png", (mask * 255).astype(np.uint8))
        mask = self.guided_filter(img, mask)

        # cv2.imwrite("test_videos/guided_filter_mask.png", (mask * 255).astype(np.uint8))
        mask_small = cv2.resize(mask, (16, 16))
        # print("mask_small:", mask_small)
        mask_small_vec = mask_small.reshape(256, 1)
        # print("mask_small_vec:", mask_small_vec)

        self.mask_list.append(mask)
        self.mask_list_small.append(mask_small_vec)
        m = len(self.mask_list_small)
        if m > 4:
            self.mask_list_small = self.mask_list_small[-4:]
            self.mask_list = self.mask_list[-4:]
        n = len(self.mask_list_small)
        
        At = np.matrix(np.array(self.mask_list_small).reshape(n, 256))
        # print("At:", At[-1,:])
        lamb = 1.0
        mask_small_cv = mask_small_vec.copy()
        
        I = np.matrix(np.eye(n))
        XTX = At * At.T
        # a = np.linalg.pinv(XTX + lamb * I)
        # b = a * At.T
        # c = b * mask_small_cv
        # print(XTX.shape)
        # print(I.shape)
        # print(mask_small_cv.shape)
        alpha = (XTX + lamb * I).I * At * mask_small_cv 
        # alpha = (At * A + lamb * np.eye(n).astype(np.float32)) * At * mask_small_cv
        out_mask = np.zeros((mask.shape[0], mask.shape[1])).astype(np.float32)
        # print(alpha)
        for i in range(n):
            out_mask += alpha[i, 0] * self.mask_list[i]

        out_mask = cv2.GaussianBlur(out_mask, (3, 3), 1.5)
        out_mask = cv2.resize(out_mask, (img.shape[1], img.shape[0]))
        # out_mask[out_mask >= 0.5] = 1.0
        # out_mask[out_mask < 0.5] = 0.0
        return out_mask


class seg_mask_filter(object):
    """docstring for seg_mask_filter"""
    def __init__(self):
        super(seg_mask_filter, self).__init__()
        self.alpha_pre = None
        self._epsilon = 1e-8

    def reset(self):
        self.alpha_pre = None

    def __call__(self, alpha_t):
        if self.alpha_pre is not None:
            p = 1 + (alpha_t * np.log(alpha_t + self._epsilon) + (1 - alpha_t) * np.log(1 - alpha_t + self._epsilon)) / np.log(2.0)
            p = np.clip(p, 0., 1.)
            p = 1 - np.square(1 - p)
            alpha_hat = alpha_t * p + self.alpha_pre * (1 - p)
            alpha = 0.9 * alpha_hat + 0.1 * alpha_t
            self.alpha_pre = alpha
        else:
            alpha = alpha_t
            self.alpha_pre = alpha_t
        return alpha


        