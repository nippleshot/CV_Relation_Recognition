import os
import yaml
import torch
from model import Classifier
from dataset import VRD


class Container:

    def __init__(self, model, dataset, cfg):
        self.model = model
        self.dataset = dataset

        # use gpu
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # hyper-parameters
        self.lr_init = cfg['train_lr']
        self.lr_adjust_rate = cfg['train_lr_adjust_rate']
        self.lr_adjust_freq = cfg['train_lr_adjust_freq']
        self.epoch_num = cfg['train_epoch_num']
        self.save_freq = cfg['train_save_freq']
        self.batch_size = cfg['train_batch_size']
        self.print_freq = cfg['train_print_freq']

        # output path
        self.checkpoint_root = cfg['checkpoint_root']

    def train(self):
        data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        batch_num = len(data_loader)

        self.model.to(self.device)
        self.model.train()
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.lr_init)
        optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=self.lr_init)
        loss_func = torch.nn.BCELoss()

        curr_epoch = 0
        while curr_epoch < self.epoch_num:
            for batch_id, batch in enumerate(data_loader):
                '''
                batch = [
                    [histograms[] batch_size 개], 
                    [directions[] batch_size 개], 
                    [ratios[] batch_size 개], 
                    [sub_ids[] batch_size 개], 
                    [obj_ids[] batch_size 개], 
                    [labels[] batch_size 개]
                ]
                '''
                input_feature, labels = batch
                input_feature = input_feature.type(torch.FloatTensor).to(self.device)
                labels = labels.type(torch.FloatTensor).to(self.device)

                predicted = self.model(input_feature)
                loss = loss_func(predicted, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.__print_info__(curr_epoch, batch_id, batch_num, loss.data.item())

            curr_epoch += 1
            self.__save_checkpoint__(curr_epoch)
            self.__adjust_lr__(curr_epoch)

    def __save_checkpoint__(self, curr_epoch):
        if not os.path.exists(self.checkpoint_root):
            os.makedirs(self.checkpoint_root)

        if curr_epoch % self.save_freq == 0:
            checkpoint_path = os.path.join(self.checkpoint_root, 'model_%d.pkl' % curr_epoch)
            torch.save(self.model.state_dict(), checkpoint_path)
            print('============================== Checkpoint saved at %s ==============================' % checkpoint_path)

    def __adjust_lr__(self, curr_epoch):
        lr_curr = self.lr_init * (self.lr_adjust_rate ** int(curr_epoch / self.lr_adjust_freq))
        self.optimizer = torch.optim.SGD([{'params': self.model.parameters()}], lr=lr_curr)
        if lr_curr != self.lr_init:
            print('\t Current adjusted_lr = %f' % lr_curr)

    def __print_info__(self, epoch, itr, itr_num, loss):
        if itr % self.print_freq == 0:
            print('[epoch %s][%s/%s] loss: %.4f' % (str(epoch+1).rjust(2), str(itr+1).rjust(4), str(itr_num), loss))


if __name__ == '__main__':
    # cfg_path = '/content/drive/MyDrive/NewRelationTask/config.yaml'
    cfg_path = 'config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    train_set = VRD(cfg['data_root'], cfg['train_split'], cfg['cache_root'])
    model = Classifier(train_set.input_feature_len, train_set.pre_category_num())
    container = Container(model, train_set, cfg)
    container.train()
