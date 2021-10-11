# coding: UTF-8
import os
import pickle
import numpy as np
from tqdm import tqdm
import yaml


def evaluation(gt_labels, predictions):
    """
    计算recall
    注意：gt_labels和predictions尺寸相同，第i行对应第i个样本
    :param gt_labels:   所有测试样本的关系标注，      numpy二值矩阵， N x C，N为样本总数，C为关系类别数
    :param predictions: 所有测试样本的关系预测置信度， numpy浮点型矩阵，N x C，N为样本总数，C为关系类别数
    :return: recall
    """
    if predictions.shape[0] != gt_labels.shape[0]:
        print('The length of predictions is supposed to be the same as that of gt_labels.')
        raise ValueError

    gt_cnt = 0
    tp_cnt = 0
    for i in range(gt_labels.shape[0]):
        gt_label = gt_labels[i]
        k = int(sum(gt_label))
        gt_cnt += k

        prediction = predictions[i]
        top_k_ids = np.argsort(prediction)[::-1][:k]
        pr_label = np.zeros(gt_label.shape[0])
        pr_label[top_k_ids] = 1
        tp_cnt += sum(pr_label * gt_label)

    print('\t Recall: %.4f \n' % (tp_cnt * 1.0 / gt_cnt))


def get_label(model_ver, split, cache_root, dataset_root):
    if model_ver == 1:
        os_path = os.path.join(cache_root, '%s_labels.bin' % split)
        with open(os_path, 'rb') as f:
            gt_labels = pickle.load(f)
        return gt_labels
    elif model_ver == 2:
        imgs_path = os.path.join(dataset_root, '%s_images' % split)
        file_list = dict()
        for root, dir, files in os.walk(imgs_path):
            for file in files:
                file_list[file] = os.path.join(root, file)

        label_builder = []
        for file_name in tqdm(sorted(file_list.keys())):
            os_path = os.path.join(cache_root, 'cropped_resized', '%s_labels_img_%s.bin' % (split, file_name.rstrip('.jpg')))

            with open(os_path, 'rb') as f:
                label_builder = label_builder + pickle.load(f)

        return np.array(label_builder)
    else:
        print("ARGUMENT_ERROR : model_ver should be 1 or 2")
        raise ValueError

if __name__ == '__main__':
    # cfg_path = '/content/drive/MyDrive/NewRelationTask/config.yaml'
    cfg_path = 'config.yaml'

    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    model_version = cfg['model_ver']
    split = cfg['test_split']
    if model_version == 2:
        if split.find('checking') != -1:
            split = 'checking'

    gt_labels = get_label(model_version, split, cfg['cache_root'], cfg['data_root'])

    gap = cfg['train_save_freq']
    end = cfg['test_epoch']
    for epoch_num in range(gap, end + gap, gap):
        if model_version == 1:
            prediction_path = os.path.join(cfg['output_root'],
                                       'model_%d.pkl_%s_predictions.bin' % (epoch_num, split))
        elif model_version == 2:
            prediction_path = os.path.join(cfg['output_root'],
                                       'model2_%d.pkl_%s_predictions.bin' % (epoch_num, split))
        else:
            raise ValueError

        with open(prediction_path, 'rb') as f:
            predictions = pickle.load(f)

        if model_version == 1:
            print('model_%d.pkl result : ' % epoch_num)
        elif model_version == 2:
            print('model2_%d.pkl result : ' % epoch_num)
        evaluation(gt_labels, predictions)