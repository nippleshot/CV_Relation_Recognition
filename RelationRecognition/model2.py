import numpy as np
from torch import nn
from torchvision import models
import json
import os
import torch
import yaml
import feature2



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Classifier2(nn.Module):
    def __init__(self, output_classes, checkpoint_path=None):
        super(Classifier2, self).__init__()
        fc_input_size = 2048 + 2048 + 32
        hidden_size = int((fc_input_size + output_classes) * (2/3))

        # subject & object 전용 CNN (input image size : 224*224*3)
        self.my_resnet152 = models.resnet152(pretrained=True)
        for param in self.my_resnet152.parameters():
            param.requires_grad = False
        self.my_resnet152.fc = Identity()

        # Interaction pattern 전용 CNN (input image size : 224*224*2)
        self.interaction_conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=(5, 5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(5, 5), stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), # Global Average Pooling
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_classes)
        )
        self.sigmoid = nn.Sigmoid()
        self.initialize_weights()

        if checkpoint_path is not None:
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                self.load_state_dict(checkpoint)
                print('Checkpoint loaded from %s' % checkpoint_path)
            else:
                print('Attention: %s not exist' % checkpoint_path)

    # 참고 : https://eyeofneedle.tistory.com/21
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, sub, obj, pred):
        sub_feature = self.my_resnet152(sub)       # torch.Size([batch_size, 2048])
        obj_feature = self.my_resnet152(obj)       # torch.Size([batch_size, 2048])
        pred_feature = self.interaction_conv(pred) # torch.Size([batch_size, 32])
        x = self.classifier(torch.cat((sub_feature, obj_feature, pred_feature), 1))
        return self.sigmoid(x)


def check_obj_feature_size():
    '''
    모델 준비
    '''
    my_resnet152 = models.resnet152(pretrained=True)
    for param in my_resnet152.parameters():
        param.requires_grad = False
    # my_resnet152.avgpool = Identity()
    my_resnet152.fc = Identity()
    print(my_resnet152)

    '''
    이미지 224*224*3 으로 준비
    '''
    cfg_path = 'config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # {"predicate": [3], "subject": {"category": 0, "bbox": [230, 72, 415, 661]}, "object": {"category": 20, "bbox": [442, 292, 601, 433]}},
    imgs_path = os.path.join(cfg['data_root'], 'train_images', '000001.jpg')
    sub_region = [230, 72, 415, 661]
    obj_region = [442, 292, 601, 433]

    sub_img, obj_img = feature2.crop_and_resize_Img(imgs_path, sub_region, obj_region, (224, 224))

    input_img = torch.from_numpy(sub_img)
    input_img = input_img.permute(2, 0, 1)
    input_img = torch.stack((input_img, input_img, input_img, input_img))
    # input_img = input_img.unsqueeze(dim=0)

    feature = my_resnet152(input_img.float())
    print("sub_img feature size ==> " + str(feature.shape))


def check_pred_feature_size():
    interaction_conv = nn.Sequential(
        nn.Conv2d(in_channels=2, out_channels=64, kernel_size=(5, 5), stride=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(5, 5), stride=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
        nn.Flatten()
    )
    print(interaction_conv)

    cfg_path = 'config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)


    # {"predicate": [3], "subject": {"category": 0, "bbox": [230, 72, 415, 661]}, "object": {"category": 20, "bbox": [442, 292, 601, 433]}},
    imgs_path = os.path.join(cfg['data_root'], 'train_images', '000001.jpg')
    sub_region = [230, 72, 415, 661]
    obj_region = [442, 292, 601, 433]

    pred_img = feature2.make_interaction_pattern(imgs_path, sub_region, obj_region, (224, 224))
    input_img = torch.from_numpy(pred_img)
    input_img = input_img.permute(2, 0, 1)
    input_img = torch.stack((input_img, input_img, input_img, input_img))
    print(str(input_img.shape))
    # input_img = input_img.unsqueeze(dim=0)

    feature = interaction_conv(input_img.float())
    print("pred_img feature size ==> " + str(feature.shape))

if __name__ == '__main__':
    check_obj_feature_size()
    # check_pred_feature_size()

