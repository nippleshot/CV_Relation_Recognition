import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Classifier(nn.Module):

    def __init__(self, feature_len, pred_category_num, checkpoint_path=None):
        super(Classifier, self).__init__()

        self.conv_1 = nn.Conv1d(1, 64, 1)
        self.maxpool1 = nn.MaxPool1d(2)
        self.conv_2 = nn.Conv1d(64, 128, 1)
        self.maxpool2 = nn.MaxPool1d(2)
        self.conv_3 = nn.Conv1d(128, 1024, 1)
        self.maxpool3 = nn.MaxPool1d(2)

        self.fc_1 = nn.Linear(1024*552, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256, pred_category_num)



        if checkpoint_path is not None:
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                self.load_state_dict(checkpoint)
                print('Checkpoint loaded from %s' % checkpoint_path)
            else:
                print('Attention: %s not exist' % checkpoint_path)

    def forward(self, features, subs, objs, direction, ratios):


        sub_obj_concat = torch.cat((subs, objs), dim=1)
        sub_obj_dir_concat = torch.cat((sub_obj_concat, direction), dim=1)
        all_concat = torch.cat((sub_obj_dir_concat, features), dim=1)
        all_concat_ratio = torch.cat((all_concat, ratios), dim=1)
        all_concat_ratio = all_concat_ratio.unsqueeze(1)
        #print(all_concat_ratio.shape)

        x = self.conv_1(all_concat_ratio)
        x = self.maxpool1(x)
        x = self.conv_2(x)
        x = self.maxpool2(x)
        x = self.conv_3(x)
        #print(x.shape)


        x = x.view(-1, 1024*552)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.sigmoid(self.fc_3(x))

        return x
