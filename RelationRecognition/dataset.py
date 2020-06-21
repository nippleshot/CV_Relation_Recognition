# coding: UTF-8
import os
import time
import json
import pickle

import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from feature import FeatureExtractor
from helper import calculate_helper


class VRD(Dataset):
    def __init__(self, dataset_root, split, cache_root='cache'):

        self.dataset_root = dataset_root
        self.is_testing = split == 'test'
        self.pre_categories = json.load(open(os.path.join(dataset_root, 'predicates.json')))
        self.obj_categories = json.load(open(os.path.join(dataset_root, 'objects.json')))

        if not os.path.exists(cache_root):
            os.makedirs(cache_root)

        if split == 'test':
            self.features, self.gt_labels, self.gt_subject_ids, self.gt_object_ids, self.gt_directions = self.__prepare_testing_data__(cache_root)
        elif split == 'train' or split == 'val':
            self.features, self.gt_labels, self.gt_subject_ids, self.gt_object_ids, self.gt_directions, self.gt_bbox_ratios = self.__prepare_training_data__(split, cache_root)
        else:
            print('split in [train, val, test]')
            raise ValueError

        self.feature_len = self.features.shape[1]

    def pre_category_num(self):
        return len(self.pre_categories)

    def obj_category_num(self):
        return len(self.obj_categories)

    def __prepare_training_data__(self, split, cache_root):
        """
        准备特征文件和标签文件
        特征文件包含一个numpy浮点型二维矩阵，N x L，N为样本总数，L为特征长度
        标签文件包含一个numpy二值型二维矩阵，N x C，N为样本总数，C为关系类别数
        :param split: train or val
        :param cache_root: save root
        :return: features, labels
        """
        feature_path = os.path.join(cache_root, '%s_features.bin' % split)
        label_path = os.path.join(cache_root, '%s_labels.bin' % split)
        sub_path = os.path.join(cache_root, '%s_subs.bin' % split)
        obj_path = os.path.join(cache_root, '%s_objs.bin' % split)
        dir_path = os.path.join(cache_root, '%s_dir.bin' % split)
        ratio_path = os.path.join(cache_root, '%s_ratio.bin' % split)

        if not os.path.exists(feature_path) or not os.path.exists(label_path):
            print('Extracting features for %s set ...' % split)
            time.sleep(2)

            imgs_path = os.path.join(self.dataset_root, '%s_images' % split)
            ano_path = os.path.join(self.dataset_root, 'annotations_%s.json' % split)

            features_builder = []
            gt_labels_builder = []
            sub_id_builder = []
            obj_id_builder = []
            direction_builder = []
            ratio_builder = []

            feature_extractor = FeatureExtractor()
            with open(ano_path, 'r') as f:
                annotation_all = json.load(f)
            file_list = dict()
            for root, dir, files in os.walk(imgs_path):
                for file in files:
                    file_list[file] = os.path.join(root, file)
            for file in tqdm(sorted(file_list.keys())):
                ano = annotation_all[file]
                samples_info = []
                labels = []

                subjects = []
                objects = []
                helper = calculate_helper()
                directions = []
                ratios = []

                for sample in ano:
                    gt_predicates = sample['predicate']
                    gt_object_id = sample['object']['category']
                    gt_object_loc = sample['object']['bbox']
                    gt_subject_id = sample['subject']['category']
                    gt_subject_loc = sample['subject']['bbox']
                    samples_info.append(gt_subject_loc + [gt_subject_id] + gt_object_loc + [gt_object_id])

                    sub_bbox_center = helper.cal_bbox_center(gt_subject_loc[0], gt_subject_loc[1], gt_subject_loc[2], gt_subject_loc[3])
                    obj_bbox_center = helper.cal_bbox_center(gt_object_loc[0],gt_object_loc[1], gt_object_loc[2], gt_object_loc[3])
                    dir_compass = helper.cal_sub2obj_direction(sub_bbox_center[0], obj_bbox_center[0], sub_bbox_center[1], sub_bbox_center[1])
                    dir_one_hot = helper.dir_convert2_one_hot(dir_compass)
                    # bbox_ratio = helper.cal_bbox_WnH(gt_subject_loc, gt_object_loc, 0)
                    bbox_ratio_with_Area = helper.cal_bbox_WnH(gt_subject_loc, gt_object_loc, 1)


                    # sub_one_hot = np.zeros(self.obj_category_num())
                    # obj_one_hot = np.zeros(self.obj_category_num())
                    # sub_one_hot[gt_subject_id] = 1
                    # obj_one_hot[gt_object_id] = 1

                    subjects.append([gt_subject_id])
                    objects.append([gt_object_id])
                    directions.append(dir_one_hot.tolist())
                    ratios.append(bbox_ratio_with_Area.tolist())

                    # subjects.append(sub_one_hot.tolist())
                    # objects.append(obj_one_hot.tolist())
                    # directions.append(dir_one_hot.tolist())

                    predicates = np.zeros(self.pre_category_num())
                    for p in gt_predicates:
                        predicates[p] = 1
                    labels.append(predicates.tolist())
                feature = feature_extractor.extract_features(cv2.imread(file_list[file]), samples_info)
                features_builder = features_builder + feature.tolist()
                gt_labels_builder = gt_labels_builder + labels

                sub_id_builder = sub_id_builder + subjects
                obj_id_builder = obj_id_builder + objects
                direction_builder = direction_builder + directions
                ratio_builder = ratio_builder + ratios

            features = np.array(features_builder)
            gt_labels = np.array(gt_labels_builder)
            gt_subject_ids = np.array(sub_id_builder)
            gt_object_ids = np.array(obj_id_builder)
            gt_directions = np.array(direction_builder)
            gt_bbox_ratios = np.array(ratio_builder)

            with open(feature_path, 'wb') as fw:
                pickle.dump(features, fw)
            with open(label_path, 'wb') as fw:
                pickle.dump(gt_labels, fw)
            with open(sub_path, 'wb') as fw:
                pickle.dump(gt_subject_ids, fw)
            with open(obj_path, 'wb') as fw:
                pickle.dump(gt_object_ids, fw)
            with open(dir_path, 'wb') as fw:
                pickle.dump(gt_directions, fw)
            with open(ratio_path, 'wb') as fw:
                pickle.dump(gt_bbox_ratios, fw)

        else:
            print('Loading data ...')
            with open(feature_path, 'rb') as f:
                features = pickle.load(f)
            with open(label_path, 'rb') as f:
                gt_labels = pickle.load(f)
            with open(sub_path, 'rb') as f:
                gt_subject_ids = pickle.load(f)
            with open(obj_path, 'rb') as f:
                gt_object_ids = pickle.load(f)
            with open(dir_path, 'rb') as f:
                gt_directions = pickle.load(f)
            with open(ratio_path, 'rb') as f:
                gt_bbox_ratios = pickle.load(f)


        return features, gt_labels, gt_subject_ids, gt_object_ids, gt_directions, gt_bbox_ratios

    def __prepare_testing_data__(self, cache_root):
        """
        准备特征文件
        特征文件包含一个numpy浮点型二维矩阵，N x L，N为样本总数，L为特征长度
        :param cache_root: save root
        :return: features, labels=None
        """
        feature_path = os.path.join(cache_root, 'test_features.bin')
        sub_path = os.path.join(cache_root, 'test_subs.bin')
        obj_path = os.path.join(cache_root, 'test_objs.bin')
        dir_path = os.path.join(cache_root, 'test_dir.bin')
        ratio_path = os.path.join(cache_root, 'test_ratio.bin')
        if not os.path.exists(feature_path):
            print('Extracting features for test set ...')
            time.sleep(2)

            imgs_path = os.path.join(self.dataset_root, 'test_images')
            ano_path = os.path.join(self.dataset_root, 'annotations_test_so.json')
            features_builder = []

            sub_id_builder = []
            obj_id_builder = []

            helper = calculate_helper()
            direction_builder = []
            ratio_builder = []

            subjects = []
            objects = []
            directions = []
            ratios = []

            feature_extractor = FeatureExtractor()
            with open(ano_path, 'r') as f:
                annotation_all = json.load(f)
            file_list = dict()
            for root, dir, files in os.walk(imgs_path):
                for file in files:
                    file_list[file] = os.path.join(root, file)
            for file in tqdm(sorted(file_list.keys())):
                ano = annotation_all[file]
                samples_info = []
                for sample in ano:
                    gt_object_id = sample['object']['category']
                    gt_object_loc = sample['object']['bbox']
                    gt_subject_id = sample['subject']['category']
                    gt_subject_loc = sample['subject']['bbox']
                    samples_info.append(gt_subject_loc + [gt_subject_id] + gt_object_loc + [gt_object_id])

                    sub_bbox_center = helper.cal_bbox_center(gt_subject_loc[0], gt_subject_loc[1], gt_subject_loc[2],
                                                             gt_subject_loc[3])
                    obj_bbox_center = helper.cal_bbox_center(gt_object_loc[0], gt_object_loc[1], gt_object_loc[2],
                                                             gt_object_loc[3])
                    dir_compass = helper.cal_sub2obj_direction(sub_bbox_center[0], obj_bbox_center[0],
                                                               sub_bbox_center[1], sub_bbox_center[1])
                    dir_one_hot = helper.dir_convert2_one_hot(dir_compass)
                    bbox_ratio_with_Area = helper.cal_bbox_WnH(gt_subject_loc, gt_object_loc, 1)

                    subjects.append([gt_subject_id])
                    objects.append([gt_object_id])
                    directions.append(dir_one_hot.tolist())
                    ratios.append(bbox_ratio_with_Area.tolist())


                feature = feature_extractor.extract_features(cv2.imread(file_list[file]), samples_info)
                features_builder = features_builder + feature.tolist()
                sub_id_builder = sub_id_builder + subjects
                obj_id_builder = obj_id_builder + objects
                direction_builder = direction_builder + directions
                ratio_builder = ratio_builder + ratios

            features = np.array(features_builder)
            gt_subject_ids = np.array(sub_id_builder)
            gt_object_ids = np.array(obj_id_builder)
            gt_directions = np.array(direction_builder)
            gt_bbox_ratios = np.array(ratio_builder)

            with open(feature_path, 'wb') as fw:
                pickle.dump(features, fw)
            with open(sub_path, 'wb') as fw:
                pickle.dump(gt_subject_ids, fw)
            with open(obj_path, 'wb') as fw:
                pickle.dump(gt_object_ids, fw)
            with open(dir_path, 'wb') as fw:
                pickle.dump(gt_directions, fw)
            with open(ratio_path, 'wb') as fw:
                pickle.dump(gt_bbox_ratios, fw)

        else:
            print('Loading data ...')
            with open(feature_path, 'rb') as f:
                features = pickle.load(f)
            with open(sub_path, 'rb') as f:
                gt_subject_ids = pickle.load(f)
            with open(obj_path, 'rb') as f:
                gt_object_ids = pickle.load(f)
            with open(dir_path, 'rb') as f:
                gt_directions = pickle.load(f)
            with open(ratio_path, 'rb') as f:
                gt_bbox_ratios = pickle.load(f)


        return features, None, gt_subject_ids, gt_object_ids, gt_directions, gt_bbox_ratios

    def __getitem__(self, item):
        if self.is_testing:
            return self.features[item], 0, self.gt_subject_ids[item], self.gt_object_ids[item], self.gt_directions[item], self.gt_bbox_ratios[item]
        else:
            return self.features[item], self.gt_labels[item], self.gt_subject_ids[item], self.gt_object_ids[item], self.gt_directions[item], self.gt_bbox_ratios[item]

    def __len__(self):
        return self.features.shape[0]

    def len(self):
        return self.features.shape[0]


