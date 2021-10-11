# coding: UTF-8
import os
import time
import json
import pickle

import torch
import yaml

import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from feature import FeatureExtractor

class VRD(Dataset):
    def __init__(self, dataset_root, split, cache_root='cache'):

        self.dataset_root = dataset_root
        self.is_testing = split.find('test') != -1
        self.pre_categories = json.load(open(os.path.join(dataset_root, 'predicates.json')))
        self.obj_categories = json.load(open(os.path.join(dataset_root, 'objects.json')))

        if not os.path.exists(cache_root):
            os.makedirs(cache_root)

        if split == 'test':
            self.total_pic_inputFeature = self.__prepare_data__(split, cache_root, 1)
        elif split == 'train' or split == 'val':
            self.total_pic_inputFeature, self.total_pic_labels = self.__prepare_data__(split, cache_root, 0)
        elif split == 'checking_train' or split == 'checking_val':
            self.total_pic_inputFeature, self.total_pic_labels = self.__prepare_data__(split, cache_root, 0)
        else:
            print('split in [train, val, test, checking_test, checking_train]')
            raise ValueError

        self.input_feature_len = self.total_pic_inputFeature.shape[1]

    def pre_category_num(self):
        return len(self.pre_categories)

    def obj_category_num(self):
        return len(self.obj_categories)
    
    '''
    mode == 0 : prepare training data
    mode == 1 : prepare testing data
    '''
    def __prepare_data__(self, split, cache_root, mode):
        
        histogram_path = os.path.join(cache_root, '%s_histograms.bin' % split)
        dir_path = os.path.join(cache_root, '%s_dir.bin' % split)
        ratio_path = os.path.join(cache_root, '%s_ratio.bin' % split)
        sub_path = os.path.join(cache_root, '%s_subs.bin' % split)
        obj_path = os.path.join(cache_root, '%s_objs.bin' % split)

        filepath_list = [histogram_path, dir_path, ratio_path, sub_path, obj_path]
        
        if mode == 0:
            label_path = os.path.join(cache_root, '%s_labels.bin' % split)
            filepath_list = filepath_list + [label_path]
            
        file_not_exist = False
        for filepath in filepath_list:
            if not os.path.exists(filepath):
                file_not_exist = True

        if file_not_exist:
            print('Extracting features for %s set ...' % split)
            time.sleep(2)
            
            if split.find('checking') != -1:
                split = 'checking'
            imgs_path = os.path.join(self.dataset_root, '%s_images' % split)
            ano_path = os.path.join(self.dataset_root, 'annotations_%s.json' % split)
            
            # "여러개" 사진에 대한 여러 훈련 데이터 
            histogram_builder = []
            direction_builder = []
            ratio_builder = []
            sub_id_builder = []
            obj_id_builder = []
            if mode == 0:
                gt_labels_builder = []

            feature_extractor = FeatureExtractor()

            # annotations 파일 읽어드려서 annotation_all 작성
            with open(ano_path, 'r') as f:
                annotation_all = json.load(f)

            file_list = dict()
            # file_list = {'file name': file path, ... }
            for root, dir, files in os.walk(imgs_path):
                for file in files:
                    file_list[file] = os.path.join(root, file)

            # file_list의 'file name' 순서대로 하나씩 선택
            for file in tqdm(sorted(file_list.keys())):
                # ano =  [{
                #   "predicate": [3],
                #   "subject": {"category": 0, "bbox": [230, 72, 415, 661]},
                #   "object": {"category": 20, "bbox": [442, 292, 601, 433]}
                #   }, ...]
                ano = annotation_all[file]

                """
                * 이 list들 에다가 "한개" 사진에 대한 여러 훈련 데이터들을 append 할 예정임
                samples_infos 형태 : [[],  [],  ...] -> 각 [] 형태  = [sub_color_hist, sub_grad_hist, obj_color_hist, obj_grad_hist] (총 길이 : 2184)
                directions 형태    : [[],  [],  ...] -> 각 [] 형태  = [one_hot] (총 길이 : 16)
                ratios 형태        : [[],  [],  ...] -> 각 [] 형태  = [one_hot, float value] (총 길이 : 9)
                sub_ids 형태       : [int, int, ...]
                obj_ids 형태       : [int, int, ...]
                labels 형태        : [[],  [],  ...] -> 각 [] 형태  = [one_hot] (총 길이 : 70)
                """
                samples_infos = [] 
                directions = []    
                ratios = []        
                sub_ids = []       
                obj_ids = []
                if mode == 0:
                    labels = []        

                for sample in ano:
                    # "한개" 사진의 "한개" 훈련 데이터의 data parsing
                    gt_predicates = sample['predicate']
                    gt_object_id = sample['object']['category']     # int 형
                    gt_object_loc = sample['object']['bbox']        # list 형
                    gt_subject_id = sample['subject']['category']   # int 형
                    gt_subject_loc = sample['subject']['bbox']      # list 형

                    # "한개" 사진의 "한개" 훈련 데이터의 subject와 object의 방향 정보 one_hot 형태로 획득
                    sub_bbox_center = feature_extractor.cal_bbox_center(gt_subject_loc[0], gt_subject_loc[1], gt_subject_loc[2], gt_subject_loc[3])
                    obj_bbox_center = feature_extractor.cal_bbox_center(gt_object_loc[0],gt_object_loc[1], gt_object_loc[2], gt_object_loc[3])
                    dir_compass = feature_extractor.cal_sub2obj_direction(sub_bbox_center[0], obj_bbox_center[0], sub_bbox_center[1], obj_bbox_center[1])
                    dir_one_hot = feature_extractor.one_hot(dir_compass)

                    # "한개" 사진의 "한개" 훈련 데이터의 subject와 object bbox 비율 및 면적 정보 획득
                    bbox_ratio_with_Area = feature_extractor.cal_bbox_WnH(gt_subject_loc, gt_object_loc, add_area_ratio=1)


                    # 획득한 정보들 추가
                    samples_infos.append(gt_subject_loc + [gt_subject_id] + gt_object_loc + [gt_object_id]) # [sub_loc] + [sub_id] + [obj_loc] + [obj_id] ==> [sub_loc, sub_id, obj_loc, obj_id]
                    directions.append(dir_one_hot.tolist())
                    ratios.append(bbox_ratio_with_Area.tolist())
                    sub_ids.append([gt_subject_id])
                    obj_ids.append([gt_object_id])
                    if mode == 0:
                        predicates = np.zeros(self.pre_category_num()) # predicates를 one_hot 형태로 변경
                        for p in gt_predicates:
                            predicates[p] = 1
                        labels.append(predicates.tolist())

                # one_pic_histograms 형태 : np.array([],[],[], ...)
                one_pic_histograms = feature_extractor.extract_features(cv2.imread(file_list[file]), samples_infos)
                # one_pic_histograms.tolist() 형태 : [[],[],[], ...] + [[],[],[], ...] = [[],[],[],[],[],[], ...]
                histogram_builder = histogram_builder + one_pic_histograms.tolist()
                direction_builder = direction_builder + directions
                ratio_builder = ratio_builder + ratios
                sub_id_builder = sub_id_builder + sub_ids
                obj_id_builder = obj_id_builder + obj_ids
                if mode == 0:
                    gt_labels_builder = gt_labels_builder + labels

            total_pic_histograms = np.array(histogram_builder)
            total_pic_directions = np.array(direction_builder)
            total_pic_ratios = np.array(ratio_builder)
            total_pic_sub_ids = np.array(sub_id_builder)
            total_pic_obj_ids = np.array(obj_id_builder)
            if mode == 0:
                total_pic_labels = np.array(gt_labels_builder)

            with open(histogram_path, 'wb') as fw:
                pickle.dump(total_pic_histograms, fw)
            with open(dir_path, 'wb') as fw:
                pickle.dump(total_pic_directions, fw)
            with open(ratio_path, 'wb') as fw:
                pickle.dump(total_pic_ratios, fw)
            with open(sub_path, 'wb') as fw:
                pickle.dump(total_pic_sub_ids, fw)
            with open(obj_path, 'wb') as fw:
                pickle.dump(total_pic_obj_ids, fw)
            if mode == 0:
                with open(label_path, 'wb') as fw:
                    pickle.dump(total_pic_labels, fw)

        else:
            print('Loading data ...')
            with open(histogram_path, 'rb') as f:
                total_pic_histograms = pickle.load(f)
            with open(dir_path, 'rb') as f:
                total_pic_directions = pickle.load(f)
            with open(ratio_path, 'rb') as f:
                total_pic_ratios = pickle.load(f)
            with open(sub_path, 'rb') as f:
                total_pic_sub_ids = pickle.load(f)
            with open(obj_path, 'rb') as f:
                total_pic_obj_ids = pickle.load(f)
            if mode == 0:
                with open(label_path, 'rb') as f:
                    total_pic_labels = pickle.load(f)
        
        if mode == 0:
            return np.concatenate((total_pic_histograms, total_pic_directions, total_pic_ratios, total_pic_sub_ids, total_pic_obj_ids), axis=1), total_pic_labels 
        elif mode == 1:
            return np.concatenate((total_pic_histograms, total_pic_directions, total_pic_ratios, total_pic_sub_ids, total_pic_obj_ids), axis=1)
        else:
            print("ARGUMENT_ERROR : mode should be 0 or 1")
            raise ValueError
    
    def __getitem__(self, item):
        if self.is_testing:
            return self.total_pic_inputFeature[item]
        else:
            return self.total_pic_inputFeature[item], self.total_pic_labels[item]

    def __len__(self):
        # 준비한 train/val/train 데이터가 총 몇개 인지?
        return self.total_pic_inputFeature.shape[0] 

    def len(self):
        return self.total_pic_inputFeature.shape[0]


if __name__ == '__main__':
    cfg_path = 'config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    train_set = VRD(cfg['data_root'], cfg['train_split'], cfg['cache_root'])
    print(train_set.__getitem__(3))