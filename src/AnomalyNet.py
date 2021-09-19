import torch
import pytorch_lightning as pl
import torch.nn as nn
import torchvision.models as models
from FDFE import multiPoolPrepare, multiMaxPooling, unwrapPrepare, unwrapPool


class AnomalyNet:
    def __init__(self):
        self.patch_cnn = {
            **dict.fromkeys([(65, 65), 65, 'big'], AnomalyNet65),
            **dict.fromkeys([(33, 33), 33, 'medium'], AnomalyNet33),
            **dict.fromkeys([(17, 17), 17, 'small'], AnomalyNet17),
        }

    @classmethod
    def create(cls, model_name):
        self = cls()
        model = self.patch_cnn[model_name]
        if not model:
            raise ValueError(f'Model not found - {model_name}')
        return model()


class AnomalyNet65(pl.LightningModule):
    '''Patch-based CNN for anomaly detection. Designed to work with patches of size 65x65.
    '''
    size = 65

    def __init__(self):
        super(AnomalyNet65, self).__init__()
        self.pH = 65
        self.pW = 65
        self.multiPoolPrepare = multiPoolPrepare(self.pH, self.pW)

        self.conv1 = nn.Conv2d(3, 128, 5, 1)
        self.conv2 = nn.Conv2d(128, 128, 5, 1)
        self.conv3 = nn.Conv2d(128, 256, 5, 1)
        self.conv4 = nn.Conv2d(256, 256, 4, 1)
        self.conv5 = nn.Conv2d(256, 128, 1, 1)
        self.outChans = self.conv5.out_channels
        self.decode = nn.Linear(128, 512)
        
        self.dropout_2d = nn.Dropout2d(0.2)
        self.dropout = nn.Dropout(0.2)
        
        self.max_pool = nn.MaxPool2d(2, 2)
        self.multiMaxPooling = multiMaxPooling(2, 2, 2, 2)
        self.unwrapPrepare = unwrapPrepare()

        self.l_relu = nn.LeakyReLU(5e-3)

    def fdfe(self, x):
        '''Use Fast Dense Feature Extraction to efficiently apply 
        the patch-based CNN AnomalyNet65 on a whole image.'''

        imH = x.size(2)
        imW = x.size(3)

        unwrapPool3 = unwrapPool(self.outChans, imH / (2 * 2 * 2), imW / (2 * 2 * 2), 2, 2)
        unwrapPool2 = unwrapPool(self.outChans, imH / (2 * 2), imW / (2 * 2), 2, 2)
        unwrapPool1 = unwrapPool(self.outChans, imH / 2, imW / 2, 2, 2)

        x = self.multiPoolPrepare(x)

        x = self.l_relu(self.conv1(x))
        x = self.multiMaxPooling(x)

        x = self.l_relu(self.conv2(x))
        x = self.multiMaxPooling(x)

        x = self.l_relu(self.conv3(x))
        x = self.multiMaxPooling(x)

        x = self.l_relu(self.conv4(x))
        x = self.l_relu(self.conv5(x))

        x = self.unwrapPrepare(x)
        x = unwrapPool3(x)
        x = unwrapPool2(x)
        x = unwrapPool1(x)

        y = x.view(self.outChans, imH, imW, -1)
        y = y.permute(3, 1, 2, 0)
        y = self.l_relu(self.decode(y))
        return y

    def forward(self, x, fdfe=False):
        if fdfe:
            return self.fdfe(x)
        else:
            assert x.size(2) == self.pH and x.size(3) == self.pW, \
                f"This patch extractor only accepts input of size (b, 3, {self.pH}, {self.pW})"
            x = self.l_relu(self.conv1(x))
            x = self.max_pool(x)
            x = self.l_relu(self.conv2(x))
            x = self.max_pool(x)
            x = self.l_relu(self.conv3(x))
            x = self.max_pool(x)
            x = self.l_relu(self.conv4(x))
            x = self.l_relu(self.conv5(x))
            x = self.dropout_2d(x)
            x = x.view(-1, self.outChans)
            x = self.l_relu(self.decode(x))
            x = self.dropout(x)
            return x


class AnomalyNet33(pl.LightningModule):
    '''Patch-based CNN for anomaly detection. Designed to work with patches of size 33x33.
    '''
    size = 33

    def __init__(self):
        super(AnomalyNet33, self).__init__()
        self.pH = 33
        self.pW = 33
        self.multiPoolPrepare = multiPoolPrepare(self.pH, self.pW)

        self.conv1 = nn.Conv2d(3, 128, 5, 1)
        self.conv2 = nn.Conv2d(128, 256, 5, 1)
        self.conv3 = nn.Conv2d(256, 256, 2, 1)
        self.conv4 = nn.Conv2d(256, 128, 4, 1)
        self.outChans = self.conv4.out_channels
        self.decode = nn.Linear(128, 512)

        self.dropout_2d = nn.Dropout2d(0.2)
        self.dropout = nn.Dropout(0.2)
        
        self.max_pool = nn.MaxPool2d(2, 2)
        self.multiMaxPooling = multiMaxPooling(2, 2, 2, 2)
        self.unwrapPrepare = unwrapPrepare()

        self.l_relu = nn.LeakyReLU(5e-3)

    def fdfe(self, x):
        '''Use Fast Dense Feature Extraction to efficiently apply 
        the patch-based CNN AnomalyNet33 on a whole image.'''

        imH = x.size(2)
        imW = x.size(3)

        unwrapPool2 = unwrapPool(self.outChans, imH / (2 * 2), imW / (2 * 2), 2, 2)
        unwrapPool1 = unwrapPool(self.outChans, imH / 2, imW / 2, 2, 2)

        x = self.multiPoolPrepare(x)

        x = self.l_relu(self.conv1(x))
        x = self.multiMaxPooling(x)

        x = self.l_relu(self.conv2(x))
        x = self.multiMaxPooling(x)

        x = self.l_relu(self.conv3(x))
        x = self.l_relu(self.conv4(x))

        x = self.unwrapPrepare(x)
        x = unwrapPool2(x)
        x = unwrapPool1(x)

        y = x.view(self.outChans, imH, imW, -1)
        y = y.permute(3, 1, 2, 0)
        y = self.l_relu(self.decode(y))
        return y

    def forward(self, x, fdfe=False):
        if fdfe:
            return self.fdfe(x)
        else:
            assert x.size(2) == self.pH and x.size(3) == self.pW, \
                f"This patch extractor only accepts input of size (b, 3, {self.pH}, {self.pW})"
            x = self.l_relu(self.conv1(x))
            x = self.max_pool(x)
            x = self.l_relu(self.conv2(x))
            x = self.max_pool(x)
            x = self.l_relu(self.conv3(x))
            x = self.l_relu(self.conv4(x))
            x = self.dropout_2d(x)
            x = x.view(-1, self.outChans)
            x = self.l_relu(self.decode(x))
            x = self.dropout(x)
            return x


class AnomalyNet17(pl.LightningModule):
    '''Patch-based CNN for anomaly detection. Designed to work with patches of size 17x17
    '''
    size = 17

    def __init__(self):
        super(AnomalyNet17, self).__init__()
        self.pH = 17
        self.pW = 17
        self.multiPoolPrepare = multiPoolPrepare(self.pH, self.pW)

        self.conv1 = nn.Conv2d(3, 128, 6, 1)
        self.conv2 = nn.Conv2d(128, 256, 5, 1)
        self.conv3 = nn.Conv2d(256, 256, 5, 1)
        self.conv4 = nn.Conv2d(256, 128, 4, 1)
        self.outChans = self.conv4.out_channels
        self.decode = nn.Linear(128, 512)

        self.dropout_2d = nn.Dropout2d(0.2)
        self.dropout = nn.Dropout(0.2)

        self.l_relu = nn.LeakyReLU(5e-3)

    def fdfe(self, x):
        '''Use Fast Dense Feature Extraction to efficiently apply 
        the patch-based CNN AnomalyNet17 on a whole image.'''

        imH = x.size(2)
        imW = x.size(3)

        x = self.multiPoolPrepare(x)

        x = self.l_relu(self.conv1(x))
        x = self.l_relu(self.conv2(x))
        x = self.l_relu(self.conv3(x))
        x = self.l_relu(self.conv4(x))

        y = x.view(self.outChans, imH, imW, -1)
        y = y.permute(3, 1, 2, 0)
        y = self.l_relu(self.decode(y))
        return y

    def forward(self, x, fdfe=False):
        if fdfe:
            return self.fdfe(x)
        else:
            assert x.size(2) == self.pH and x.size(3) == self.pW, \
                f"This patch extractor only accepts input of size (b, 3, {self.pH}, {self.pW})"
            x = self.l_relu(self.conv1(x))
            x = self.l_relu(self.conv2(x))
            x = self.l_relu(self.conv3(x))
            x = self.l_relu(self.conv4(x))
            x = self.dropout_2d(x)
            x = x.view(-1, self.outChans)
            x = self.l_relu(self.decode(x))
            x = self.dropout(x)
            return x


if __name__ == '__main__':

    pH = 33
    pW = 33

    # Pretrained network for knowledge distillation
    resnet18 = models.resnet18(pretrained=True)
    resnet18 = nn.Sequential(*list(resnet18.children())[:-1])

    net = AnomalyNet.create((pH, pW))
    x = torch.rand((5, 3, pH, pW))

    y_net = net(x)
    y_resnet18 = resnet18(x)

    print(y_net.size())
    print(torch.squeeze(y_resnet18).size())
    