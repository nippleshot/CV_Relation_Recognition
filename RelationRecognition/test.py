import os
import pickle

import yaml
import torch
import numpy as np
from tqdm import tqdm
from model import Classifier
from dataset import VRD


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
                input_feature, _ = batch
                input_feature = input_feature.type(torch.FloatTensor).to(self.device)

                predicted = self.model(input_feature)
                predicted = predicted.cpu()
                predicted = predicted.squeeze(1)
                predicted = predicted.data.numpy()
                results[data_cnt: data_cnt+input_feature.shape[0]] = predicted
                data_cnt += input_feature.shape[0]
        return results


if __name__ == '__main__':
    # cfg_path = '/content/drive/MyDrive/NewRelationTask/config.yaml'
    cfg_path = 'config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    test_set = VRD(cfg['data_root'], cfg['test_split'], cfg['cache_root'])
    print("Test set 개수    : " + str(len(test_set)))
    print("Class label 개수 : " + str(test_set.pre_category_num()))

    gap = cfg['train_save_freq']
    end = cfg['test_epoch']
    for epoch_num in range(gap, end + gap, gap):
        checkpoint_path = os.path.join(cfg['checkpoint_root'], 'model_%d.pkl' % epoch_num)
        model = Classifier(test_set.input_feature_len, test_set.pre_category_num(), checkpoint_path)
        container = Container(model, test_set, cfg)

        output_root = cfg['output_root']
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        output_path = os.path.join(output_root, 'model_%d.pkl_%s_predictions.bin' % (epoch_num, cfg['test_split']))
        results = container.test()
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        print('Predictions are saved at %s' % output_path)

