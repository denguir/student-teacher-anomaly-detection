import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

from config import Config
from utils import load_model


class AnomalyResnet18(nn.Module):
    """Resnet18 extended for anomaly classification:
    Designed to tell wether an object presents
    anomalies or not"""

    def __init__(self):
        super(AnomalyResnet18, self).__init__()
        self.resnet18 = self._get_resnet18_backbone()
        self.linear = nn.Linear(512, 2)
        self.softmax = nn.Softmax(dim=1)

    def _get_resnet18_backbone(self):
        resnet18 = models.resnet18(pretrained=True)
        resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
        return resnet18

    def forward(self, x):
        x = self.resnet18(x)
        x = x.view(-1, 512)
        y = self.softmax(self.linear(x))
        return y


def load_backbone_model(dataset_name: str):
    model = AnomalyResnet18()
    model_path = Config.MODEL_PATH / dataset_name / f"resnet18.pt"
    model.load_state_dict(torch.load(model_path))
    return model


if __name__ == "__main__":
    import sys

    resnet18 = load_backbone_model(sys.argv[1])
    resnet18 = nn.Sequential(*list(resnet18.children())[:-2])
    resnet18.cuda()
    summary(resnet18, (3, 65, 65))
