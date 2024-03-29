# coding: UTF-8
import os
import time
import json
import pickle
import feature2
import yaml
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import cv2



class VRD2Helper(Dataset):
    def __init__(self, dataset_root, split, cache_root='cache'):

        self.dataset_root = dataset_root
        self.is_testing = split.find('test') != -1
        self.pre_categories = json.load(open(os.path.join(dataset_root, 'predicates.json')))
        self.obj_categories = json.load(open(os.path.join(dataset_root, 'objects.json')))

        if not os.path.exists(cache_root):
            os.makedirs(cache_root)

        if split == 'test':
            self.total_pic_sub_imgs, self.total_pic_obj_imgs, self.total_pic_pred_imgs = self.__prepare_data__(split,
                                                                                                               cache_root,
                                                                                                               1)
        elif split == 'train' or split == 'val':
            self.total_pic_sub_imgs, self.total_pic_obj_imgs, self.total_pic_pred_imgs, self.total_pic_labels = self.__prepare_data__(
                split, cache_root, 0)
        elif split == 'checking_train' or split == 'checking_val':
            self.total_pic_sub_imgs, self.total_pic_obj_imgs, self.total_pic_pred_imgs, self.total_pic_labels = self.__prepare_data__(
                split, cache_root, 0)
        else:
            print('split in [train, val, test, checking_test, checking_train]')
            raise ValueError

    def pre_category_num(self):
        return len(self.pre_categories)

    def obj_category_num(self):
        return len(self.obj_categories)

    '''
    mode == 0 : prepare training data
    mode == 1 : prepare testing data
    '''
    def __prepare_data__(self, split, cache_root, mode):
        if split.find('checking') != -1:
            split = 'checking'
        imgs_path = os.path.join(self.dataset_root, '%s_images' % split)
        ano_path = os.path.join(self.dataset_root, 'annotations_%s.json' % split)

        file_list = dict()
        # file_list = {'file name.jpg': file path, ... }
        for root, dir, files in os.walk(imgs_path):
            for file in files:
                file_list[file] = os.path.join(root, file)

        # annotations 파일 읽어드려서 annotation_all 작성
        with open(ano_path, 'r') as f:
            annotation_all = json.load(f)

        sub_img_builder = []
        obj_img_builder = []
        pred_img_builder = []
        if mode == 0:
            img_label_builder = []

        for file in tqdm(sorted(file_list.keys())):
            file_name = file.rstrip('.jpg')
            sub_img_path = os.path.join(cache_root, 'cropped_resized', '%s_subs_img_%s.bin' % (split, file_name))
            obj_img_path = os.path.join(cache_root, 'cropped_resized', '%s_objs_img_%s.bin' % (split, file_name))
            pred_img_path = os.path.join(cache_root, 'cropped_resized', '%s_preds_img_%s.bin' % (split, file_name))
            if mode == 0:
                label_path = os.path.join(cache_root, 'cropped_resized', '%s_labels_img_%s.bin' % (split, file_name))

            if not os.path.exists(sub_img_path) and not os.path.exists(obj_img_path) and not os.path.exists(
                    pred_img_path):
                print("\n[BIN FILE NOT EXIST] : Starting crop and resize %s.jpg" % file_name)
                time.sleep(2)

                ano = annotation_all[file]
                """
                * 이 list들 에다가 "한개" 사진에 대한 여러 훈련 데이터들을 append 할 예정임
                sub_imgs 형태          : [np[224*224*3], np[224*224*3], ...]
                obj_imgs 형태          : [np[224*224*3], np[224*224*3], ...]
                pred_imgs 형태         : [np[224*224*2], np[224*224*2], ...]
                img_labels 형태        : [[],  [],  ...] -> 각 [] 형태  = [one_hot] (총 길이 : 70)
                """
                sub_imgs = []
                obj_imgs = []
                pred_imgs = []
                if mode == 0:
                    img_labels = []

                for sample in ano:
                    # "한개" 사진의 "한개" 훈련 데이터의 data parsing
                    gt_predicates = sample['predicate']
                    gt_subject_loc = sample['subject']['bbox']  # list 형
                    gt_object_loc = sample['object']['bbox']  # list 형

                    # "한개" 사진의 "한개" 훈련 데이터의 sub_img, obj_img, pred_img를 획득
                    sub_img, obj_img = feature2.crop_and_resize_Img(file_list[file], gt_subject_loc, gt_object_loc,
                                                                    (224, 224))  # numpy (224, 224, 3)
                    pred_img = feature2.make_interaction_pattern(file_list[file], gt_subject_loc, gt_object_loc,
                                                                 (224, 224))  # numpy (224, 224, 2)

                    # 획득한 img들 추가
                    sub_imgs.append(sub_img)
                    obj_imgs.append(obj_img)
                    pred_imgs.append(pred_img)
                    if mode == 0:
                        predicates = np.zeros(self.pre_category_num())  # predicates를 one_hot 형태로 변경
                        for p in gt_predicates:
                            predicates[p] = 1
                        img_labels.append(predicates.tolist())

                if mode == 0:
                    save(sub_img_path, sub_imgs, obj_img_path, obj_imgs, pred_img_path, pred_imgs, 0, label_path,
                         img_labels)
                else:
                    save(sub_img_path, sub_imgs, obj_img_path, obj_imgs, pred_img_path, pred_imgs, 1)

            else:
                print("\n[BIN FILE EXIST]     : %s.jpg" % file_name)

        print("------------------------- FINISHED -------------------------")
        sub_img_builder = np.array(sub_img_builder)
        obj_img_builder = np.array(obj_img_builder)
        pred_img_builder = np.array(pred_img_builder)

        if mode == 0:
            img_label_builder = np.array(img_label_builder)
            return sub_img_builder, obj_img_builder, pred_img_builder, img_label_builder
        elif mode == 1:
            return sub_img_builder, obj_img_builder, pred_img_builder
        else:
            print("ARGUMENT_ERROR : mode should be 0 or 1")
            raise ValueError

    def __getitem__(self, item):
        if self.is_testing:
            return self.total_pic_sub_imgs[item], self.total_pic_obj_imgs[item], self.total_pic_pred_imgs[item]
        else:
            return self.total_pic_sub_imgs[item], self.total_pic_obj_imgs[item], self.total_pic_pred_imgs[item], \
                   self.total_pic_labels[item]

    def __len__(self):
        # 준비한 train/val/train 데이터가 총 몇개 인지?
        return self.total_pic_pred_imgs.shape[0]

    def len(self):
        return self.total_pic_pred_imgs.shape[0]


def save(sub_img_path, sub_imgs, obj_img_path, obj_imgs, pred_img_path, pred_imgs, mode, label_path, pic_labels):
    with open(sub_img_path, 'wb') as fw:
        pickle.dump(sub_imgs, fw)
    with open(obj_img_path, 'wb') as fw:
        pickle.dump(obj_imgs, fw)
    with open(pred_img_path, 'wb') as fw:
        pickle.dump(pred_imgs, fw)
    if mode == 0:
        with open(label_path, 'wb') as fw:
            pickle.dump(pic_labels, fw)


def concat_bin(dataset_root, cache_root, split):
    if split.find('checking') != -1:
        split = 'checking'
    imgs_path = os.path.join(dataset_root, '%s_images' % split)

    file_list = dict()
    # file_list = {'file name.jpg': file path, ... }
    for root, dir, files in os.walk(imgs_path):
        for file in files:
            file_list[file] = os.path.join(root, file)

    sub_img_builder = []
    for file in sorted(file_list.keys()):
        file_name = file.rstrip('.jpg')
        print("Loading : " + file_name)
        sub_img_path = os.path.join(cache_root, 'cropped_resized', '%s_subs_img_%s.bin' % (split, file_name))
        with open(sub_img_path, 'rb') as f:
            sub_img_builder = sub_img_builder + pickle.load(f)
    sub_total_path = os.path.join(cache_root, 'cropped_resized_total', '%s_subs_img_total.bin' % split)
    with open(sub_total_path, 'wb') as fw:
        pickle.dump(sub_img_builder, fw)
    del sub_img_builder
    print("\nDone : subs_img_total.bin\n")

    obj_img_builder = []
    for file in sorted(file_list.keys()):
        file_name = file.rstrip('.jpg')
        print("Loading : " + file_name)
        obj_img_path = os.path.join(cache_root, 'cropped_resized', '%s_objs_img_%s.bin' % (split, file_name))
        with open(obj_img_path, 'rb') as f:
            obj_img_builder = obj_img_builder + pickle.load(f)
    obj_total_path = os.path.join(cache_root, 'cropped_resized_total', '%s_objs_img_total.bin' % split)
    with open(obj_total_path, 'wb') as fw:
        pickle.dump(obj_img_builder, fw)
    del obj_img_builder
    print("\nDone : objs_img_total.bin\n")

    pred_img_builder = []
    for file in sorted(file_list.keys()):
        file_name = file.rstrip('.jpg')
        print("Loading : " + split + file_name)
        pred_img_path = os.path.join(cache_root, 'cropped_resized', '%s_preds_img_%s.bin' % (split, file_name))
        with open(pred_img_path, 'rb') as f:
            pred_img_builder = pred_img_builder + pickle.load(f)
    pred_total_path = os.path.join(cache_root, 'cropped_resized_total', '%s_preds_img_total.bin' % split)
    print("pred_img_builder length ==> " + str(len(pred_img_builder)))
    print("Start saving : preds_img_total.bin\n")
    with open(pred_total_path, 'wb') as fw:
        pickle.dump(pred_img_builder, fw)
    print("\nDone : preds_img_total.bin\n")

    img_label_builder = []
    for file in sorted(file_list.keys()):
        file_name = file.rstrip('.jpg')
        print("Loading : " + file_name)
        label_path = os.path.join(cache_root, 'cropped_resized', '%s_labels_img_%s.bin' % (split, file_name))
        with open(label_path, 'rb') as f:
            img_label_builder = img_label_builder + pickle.load(f)
    label_total_path = os.path.join(cache_root, 'cropped_resized_total', '%s_labels_img_total.bin' % split)
    with open(label_total_path, 'wb') as fw:
        pickle.dump(img_label_builder, fw)
    del img_label_builder
    print("\nDone : labels_img_total.bin\n")


def check_image(train_set, item_num):
    label = train_set.__getitem__(item_num)[3]
    for idx in range(0, len(label)):
        if label[idx] == 1:
            print("\nlabel ==> " + train_set.pre_categories[idx])

    cv2.imshow("sub", train_set.__getitem__(item_num)[0])
    cv2.imshow("obj", train_set.__getitem__(item_num)[1])

    pred = train_set.__getitem__(item_num)[2]
    imsi_channel = np.zeros((pred.shape[0], pred.shape[1], 1))
    visual_pred = np.concatenate((pred, imsi_channel), 2)
    cv2.imshow("pred", visual_pred)

    cv2.waitKey()
    cv2.destroyAllWindows()

def check_size(cache_root):
  sub_t_path = os.path.join(cache_root, 'cropped_resized_total', 'train_subs_img_total.bin')
  obj_t_path = os.path.join(cache_root, 'cropped_resized_total', 'train_objs_img_total.bin')
  pred_t_path = os.path.join(cache_root, 'cropped_resized_total', 'train_preds_img_total.bin')
  label_t_path = os.path.join(cache_root, 'cropped_resized_total', 'train_labels_img_total.bin')

  path_list = [sub_t_path, obj_t_path, pred_t_path, label_t_path]
  for path in path_list:
    with open(path, 'rb') as f :
      print(path, " length : ", len(pickle.load(f)))
    
  


if __name__ == '__main__':
    cfg_path = 'config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    check_size(cfg['cache_root'])
    # concat_bin2000(cfg['data_root'], cfg['cache_root'], cfg['train_split'])
    # concat_bin(cfg['data_root'], cfg['cache_root'], cfg['train_split'])
    # concat_bin(cfg['data_root'], cfg['cache_root'], cfg['test_split'])

    # concat_train_preds_bin(cfg['data_root'], cfg['cache_root'], cfg['train_split'])

    # train_set = VRD2Helper(cfg['data_root'], cfg['train_split'], cfg['cache_root'])
    # val_set = VRD2Helper(cfg['data_root'], cfg['test_split'], cfg['cache_root'])

    # item_num = 15
    # check_image(train_set, item_num)


