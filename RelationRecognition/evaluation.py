# coding: UTF-8
import os
import pickle
import numpy as np

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

    print('Recall: %.4f' % (tp_cnt * 1.0 / gt_cnt))


if __name__ == '__main__':
    cfg_path = 'config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    split = cfg['test_split']
    prediction_path = os.path.join(cfg['output_root'], '%s_predictions.bin' % split)
    gt_label_path = os.path.join(cfg['cache_root'], '%s_labels.bin' % split)

    with open(prediction_path, 'rb') as f:
        predictions = pickle.load(f)

    with open(gt_label_path, 'rb') as f:
        gt_labels = pickle.load(f)

    evaluation(gt_labels, predictions)
