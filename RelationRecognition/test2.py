import os
import pickle

import yaml
import torch
import numpy as np
from tqdm import tqdm
from model2 import Classifier2
from dataset2 import VRD2


class Container:

    def __init__(self, model, dataset, cfg):
        self.dataset = dataset
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = cfg['test_batch_size']

    def test(self):
        data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size)

        self.model.to(self.device)
        self.model.eval()

        data_cnt = 0
        results = np.zeros((self.dataset.len(), self.dataset.pre_category_num()))
        with torch.no_grad():
            for batch in tqdm(data_loader):
                '''
                sub_imgs size   ==> torch.Size([20, 224, 224, 3])
                obj_imgs size   ==> torch.Size([20, 224, 224, 3])
                pred_imgs size  ==> torch.Size([20, 224, 224, 2])
                img_labels size ==> torch.Size([20, 70])
                '''
                sub_imgs, obj_imgs, pred_imgs, _ = batch
                sub_imgs = sub_imgs.permute(0, 3, 1, 2).type(torch.FloatTensor).to(self.device)
                obj_imgs = obj_imgs.permute(0, 3, 1, 2).type(torch.FloatTensor).to(self.device)
                pred_imgs = pred_imgs.permute(0, 3, 1, 2).type(torch.FloatTensor).to(self.device)

                predicted = self.model(sub_imgs, obj_imgs, pred_imgs)
                predicted = predicted.cpu()
                predicted = predicted.squeeze(1)
                predicted = predicted.data.numpy()
                results[data_cnt: data_cnt+sub_imgs.shape[0]] = predicted
                data_cnt += sub_imgs.shape[0]
        return results


if __name__ == '__main__':
    # cfg_path = '/content/drive/MyDrive/NewRelationTask/config.yaml'
    cfg_path = 'config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    print("\n=========[ START : LOADING TEST DATASET FOR MODEL 2    ]=========")
    test_set = VRD2(cfg['data_root'], cfg['test_split'], True, cfg['cache_root'])
    print("=========[ FINISHED : LOADING TEST DATASET FOR MODEL 2 ]=========")
    print("Test set 개수    : " + str(len(test_set)))
    print("Class label 개수 : " + str(test_set.pre_category_num()))

    gap = cfg['train_save_freq']
    end = cfg['test_epoch']
    for epoch_num in range(gap, end + gap, gap):
        checkpoint_path = os.path.join(cfg['checkpoint_root'], 'model2_%d.pkl' % epoch_num)
        model = Classifier2(test_set.pre_category_num(), checkpoint_path)
        container = Container(model, test_set, cfg)

        output_root = cfg['output_root']
        if not os.path.exists(output_root):
            os.makedirs(output_root)

        split = cfg['test_split']
        if split.find('checking') != -1:
            split = 'checking'
        output_path = os.path.join(output_root, 'model2_%d.pkl_%s_predictions.bin' % (epoch_num, split))
        print("\n=========[ START : TESTING MODEL (model2_%d.pkl)    ]=========" % epoch_num)
        results = container.test()
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        print("=========[ FINISHED : TESTING MODEL (model2_%d.pkl) ]=========" % epoch_num)

