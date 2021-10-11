import os
import torch
from torch import nn
import torch.onnx

class Classifier(nn.Module):
    def __init__(self, input_feature, output_classes, checkpoint_path=None):
        super(Classifier, self).__init__()
        hidden_size = int((input_feature + output_classes) * (2/3))

        self.classifier = nn.Sequential(
            nn.Linear(input_feature, hidden_size),
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

    def forward(self, x):
        x = self.classifier(x)
        return self.sigmoid(x)

