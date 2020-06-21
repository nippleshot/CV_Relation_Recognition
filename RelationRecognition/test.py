import os
import pickle

import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable

from model import Classifier
from dataset import VRD


class Container:

    def __init__(self, model, dataset, cfg):
        self.dataset = dataset
        self.model = model
        self.use_gpu = cfg['use_gpu']
        self.batch_size = cfg['test_batch_size']

    def test(self):
        self.model.eval()
        features = Variable(torch.FloatTensor(1))
        sub_ids = Variable(torch.FloatTensor(1))
        obj_ids = Variable(torch.FloatTensor(1))
        directions = Variable(torch.FloatTensor(1))
        ratios = Variable(torch.FloatTensor(1))

        if self.use_gpu:
            self.model.cuda()
            features = features.cuda()
            sub_ids = sub_ids.cuda()
            obj_ids = obj_ids.cuda()
            directions = directions.cuda()
            ratios = ratios.cuda()

        data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size)

        data_cnt = 0
        results = np.zeros((self.dataset.len(), self.dataset.pre_category_num()))
        for batch in tqdm(data_loader):
            raw_features, _, raw_sub_ids, raw_obj_ids, raw_directions, raw_ratios = batch
            features.data.resize_(raw_features.size()).copy_(raw_features)

            sub_ids.data.resize_(raw_sub_ids.size()).copy_(raw_sub_ids)
            obj_ids.data.resize_(raw_obj_ids.size()).copy_(raw_obj_ids)
            directions.data.resize_(raw_directions.size()).copy_(raw_directions)
            ratios.data.resize_(raw_ratios.size()).copy_(raw_ratios)

            sub_ids = sub_ids.type(torch.cuda.FloatTensor)
            obj_ids = obj_ids.type(torch.cuda.FloatTensor)
            directions = directions.type(torch.cuda.FloatTensor)
            ratios = ratios.type(torch.cuda.FloatTensor)

            confidences = self.model(features, sub_ids, obj_ids, directions, ratios)

            if self.use_gpu:
                confidences = confidences.cpu()

            confidences = confidences.squeeze(1)
            confidences = confidences.data.numpy()
            results[data_cnt: data_cnt+raw_features.shape[0]] = confidences
            data_cnt += raw_features.shape[0]

        return results


if __name__ == '__main__':
    cfg_path = 'config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    test_set = VRD(cfg['data_root'], cfg['test_split'], cfg['cache_root'])
    checkpoint_path = os.path.join(cfg['checkpoint_root'], 'model_%d.pkl' % cfg['test_epoch'])
    model = Classifier(test_set.feature_len, test_set.pre_category_num(), checkpoint_path)
    container = Container(model, test_set, cfg)

    output_root = cfg['output_root']
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    output_path = os.path.join(output_root, '%s_predictions.bin' % cfg['test_split'])
    results = container.test()
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print('Predictions are saved at %s' % output_path)
