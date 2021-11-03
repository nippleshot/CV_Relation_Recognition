import os
import yaml
import torch
from model2 import Classifier2
from dataset2 import VRD2
# import torch_xla
# import torch_xla.core.xla_model as xm


class Container:

    def __init__(self, model, dataset, cfg):
        self.model = model
        self.dataset = dataset

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = xm.xla_device()

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
        data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
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
                sub_imgs size   ==> torch.Size([20, 224, 224, 3])
                obj_imgs size   ==> torch.Size([20, 224, 224, 3])
                pred_imgs size  ==> torch.Size([20, 224, 224, 2])
                img_labels size ==> torch.Size([20, 70])
                '''
                sub_imgs, obj_imgs, pred_imgs, img_labels = batch
                sub_imgs = sub_imgs.permute(0, 3, 1, 2).type(torch.FloatTensor).to(self.device)
                obj_imgs = obj_imgs.permute(0, 3, 1, 2).type(torch.FloatTensor).to(self.device)
                pred_imgs = pred_imgs.permute(0, 3, 1, 2).type(torch.FloatTensor).to(self.device)

                img_labels = img_labels.type(torch.FloatTensor).to(self.device)

                predicted = self.model(sub_imgs, obj_imgs, pred_imgs)
                loss = loss_func(predicted, img_labels)

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
            checkpoint_path = os.path.join(self.checkpoint_root, 'model2_%d.pkl' % curr_epoch)
            torch.save(self.model.state_dict(), checkpoint_path)
            print('\tCheckpoint saved at %s\n' % checkpoint_path)

    def __adjust_lr__(self, curr_epoch):
        lr_curr = self.lr_init * (self.lr_adjust_rate ** int(curr_epoch / self.lr_adjust_freq))
        self.optimizer = torch.optim.SGD([{'params': self.model.parameters()}], lr=lr_curr)
        if lr_curr != self.lr_init:
            print('\t Current adjusted_lr = %f' % lr_curr)

    def __print_info__(self, epoch, itr, itr_num, loss):
        if itr % self.print_freq == 0:
            print('[epoch %s][%s/%s] loss: %.4f' % (str(epoch+1).rjust(2), str(itr+1).rjust(4), str(itr_num), loss))


if __name__ == '__main__':
    cfg_path = 'config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    print("\n=========[ START : LOADING TRAIN DATASET FOR MODEL 2    ]=========")
    train_set = VRD2(cfg['data_root'], cfg['train_split'], True, cfg['cache_root'])
    print("=========[ FINISHED : LOADING TRAIN DATASET FOR MODEL 2 ]=========")
    model = Classifier2(train_set.pre_category_num())
    container = Container(model, train_set, cfg)
    print("\n=========[ START : TRAINING MODEL 2    ]=========")
    container.train()
    print("=========[ FINISHED : TRAINING MODEL 2 ]=========")