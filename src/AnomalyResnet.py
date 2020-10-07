import torch
import torchvision.models as models
import torch.nn as nn


class AnomalyResnet(nn.module):
    ''' Resnet18 extended for anomaly classification:
        Designed to tell wether an object presents
        anomalies or not'''
    def __init__(self):
        super(AnomalyResnet, self).__init__()
        self.resnet18 = self._get_resnet18_backbone()
        self.linear = nn.Linear(512, 2)
        self.softmax = nn.Softmax()
    
    def _get_resnet18_backbone(self):
        resnet18 = models.resnet18(pretrained=True)
        resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
        return resnet18

    def forward(self, x):
        x = self.resnet18(x)
        x = x.view(-1, 512)
        y = self.softmax(self.linear(x))
        return y





