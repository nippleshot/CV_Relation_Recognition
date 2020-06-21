# coding: UTF-8

import cv2
import numpy as np
from skimage import feature


class FeatureExtractor:
    def __init__(self):
        # FEATURE_DIMENSION = 2 * (256 * 3 + 324)
        self.FEATURE_DIMENSION = 2184

    def extract_features(self, img, samples):
        """
        为图片中包含的若干样本提取特征
        :param img: BGR图片（请使用opencv加载）
        :param samples: 当前图片包含的数据样本列表
                        [[  sbj_xmin, sbj_ymin, sbj_xmax, sbj_ymax, sbj_category,
                            obj_xmin, obj_ymin, obj_xmax, obj_ymax, obj_category    ]]
        :return: numpy矩阵，存储当前图片中包含的所有样本的特征
        """
        features = []
        for sample in samples:
            sub_region = [sample[0], sample[1], sample[2], sample[3]]
            obj_region = [sample[5], sample[6], sample[7], sample[8]]
            sub_f = np.concatenate((self.cal_color_hist(img, sub_region), self.cal_grad_hist(img, sub_region)))
            obj_f = np.concatenate((self.cal_color_hist(img, obj_region), self.cal_grad_hist(img, obj_region)))
            features.append(np.concatenate((sub_f, obj_f)))
        return np.array(features)

    def cal_color_hist(self, img, sample):
        height, width, _ = img.shape
        mask = np.zeros((height, width), np.uint8)
        for x in range(sample[0], sample[2] if sample[2] <= width else width):
            for y in range(sample[1], sample[3] if sample[3] <= height else height):
                mask[y][x] = 1
        b_hist = cv2.calcHist([img], [0], mask, [256], [0.0, 255.0])
        g_hist = cv2.calcHist([img], [1], mask, [256], [0.0, 255.0])
        r_hist = cv2.calcHist([img], [2], mask, [256], [0.0, 255.0])
        return self.norm(np.array(list(b_hist.flat) + list(g_hist.flat) + list(r_hist.flat)))

    def cal_grad_hist(self,img, sample):
        tar_region = cv2.resize(img[sample[1]:sample[3], sample[0]:sample[2]], (64, 64), interpolation=cv2.INTER_CUBIC)
        fd = feature.hog(tar_region, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)
        return self.norm(fd)

    def norm(self, vec):
        mx = np.max(vec)
        mn = np.min(vec)
        return (vec - mn) / (mx - mn)
