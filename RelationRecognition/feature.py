# coding: UTF-8

import cv2
import numpy as np
from skimage import feature
import math
import os
import yaml

class FeatureExtractor:
    def __init__(self):
        # FEATURE_DIMENSION = 2 * (256 * 3 + 324)
        self.FEATURE_DIMENSION = 2184
        self.COMPASS_BRACKETS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW",
                                 "NW", "NNW"]

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

    def cal_sub2obj_degree(self, sub_x, obj_x, sub_y, obj_y):
        delta_x = obj_x - sub_x
        delta_y = obj_y - sub_y
        degrees_temp = math.atan2(delta_x,delta_y)/math.pi*180

        if degrees_temp<0:
            degrees_final = 360 + degrees_temp
        else:
            degrees_final = degrees_temp

        return degrees_final

    '''
    if print_mode=0 (default) :  방향을 compass_brackets 의 index로 return 
    if print_mode=1 :  방향을 "N", "NNE", "NE" .. 형태로 return 
    '''
    def cal_sub2obj_direction(self, sub_x, obj_x, sub_y, obj_y, print_mode=0):
        degree = self.cal_sub2obj_degree(sub_x, obj_x, sub_y, obj_y)

        idx = round(degree / (360. / len(self.COMPASS_BRACKETS)))
        compass = self.COMPASS_BRACKETS[idx % len(self.COMPASS_BRACKETS)]
        if print_mode == 1:
            return compass
        return self.COMPASS_BRACKETS.index(compass) # one_hot_coding 말고 그냥 바로 사용할 수 있음

    def one_hot(self, ind):
        compass_one_hot = np.zeros(len(self.COMPASS_BRACKETS))
        compass_one_hot[ind] = 1
        return compass_one_hot

    def cal_bbox_center(self, xmin, ymin, xmax, ymax):
        coords = (xmin, xmax, ymin, ymax)
        center_x, center_y = (np.average(coords[:2]), np.average(coords[2:]))
        return int(center_x), int(center_y)

    '''
    if add_area_ratio=0 (default):  idx=8에 subject와 object 면적 차이 정보 포함 x
    if add_area_ratio=1 :  idx=8에 subject와 object 면적 차이 정보 포함
    '''
    def cal_bbox_WnH(self, subject_loc, object_loc, add_area_ratio=0):

        self.indexSearch = ["XMIN", "YMIN", "XMAX", "YMAX"]
        WnH_ratio_builder = np.zeros(len(self.indexSearch)*2)
        sub_bbox = (subject_loc[0], subject_loc[1], subject_loc[2], subject_loc[3]) #[XMIN, YMIN, XMAX, YMAX] [555, 279, 887, 680]
        obj_bbox = (object_loc[0], object_loc[1], object_loc[2], object_loc[3])     #[XMIN, YMIN, XMAX, YMAX] [6, 160, 424, 746]

        sub_x_len = sub_bbox[self.indexSearch.index("XMAX")] - sub_bbox[self.indexSearch.index("XMIN")]
        sub_y_len = sub_bbox[self.indexSearch.index("YMAX")] - sub_bbox[self.indexSearch.index("YMIN")]

        obj_x_len = obj_bbox[self.indexSearch.index("XMAX")] - obj_bbox[self.indexSearch.index("XMIN")]
        obj_y_len = obj_bbox[self.indexSearch.index("YMAX")] - obj_bbox[self.indexSearch.index("YMIN")]

        if sub_x_len > sub_y_len:
            if sub_x_len > sub_y_len*2:
                WnH_ratio_builder[0] = 1
            else:
                WnH_ratio_builder[1] = 1
        else:
            if sub_y_len > sub_x_len*2:
                WnH_ratio_builder[3] = 1
            else:
                WnH_ratio_builder[2] = 1

        if obj_x_len > obj_y_len:
            if obj_x_len > obj_y_len*2:
                WnH_ratio_builder[4] = 1
            else:
                WnH_ratio_builder[5] = 1
        else:
            if obj_y_len > obj_x_len*2:
                WnH_ratio_builder[7] = 1
            else:
                WnH_ratio_builder[6] = 1

        if add_area_ratio:
            sub2obj_area = (sub_x_len*sub_y_len)/(obj_x_len*obj_y_len)
            sub2obj_area = round(sub2obj_area, 2)
            WnH_ratio_builder = np.append(WnH_ratio_builder, np.array(sub2obj_area))

        return WnH_ratio_builder

if __name__ == '__main__':
    cfg_path = 'config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    feature_test = FeatureExtractor()
    imgs_path = os.path.join(cfg['data_root'], 'train2_images', '000001.jpg')
    img = cv2.imread(imgs_path)
    samples = [[230, 72, 415, 661, 0, 442, 292, 601, 433, 20], [376, 540, 1024, 768, 0, 4, 646, 888, 767, 51]]
    print(feature_test.extract_features(img, samples).shape)
    print(feature_test.extract_features(img, samples))
    print('-----------------------------------------------------------------')
    print(feature_test.extract_features(img, samples).tolist())
    print(len(feature_test.extract_features(img, samples).tolist()))
