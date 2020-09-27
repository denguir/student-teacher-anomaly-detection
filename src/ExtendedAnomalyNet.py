# Source:
# https://github.com/erezposner/Fast_Dense_Feature_Extraction

import torch
from torch import nn
from torchsummary import summary
from FDFE import multiPoolPrepare, multiMaxPooling, unwrapPrepare, unwrapPool


class ExtendedAnomalyNet(nn.Module):
    '''CNN that uses Fast Dense Feature Extraction to
       efficiently apply a patch-based CNN <base_net> on a whole image
    '''
    def __init__(self, base_net, pH, pW, imH, imW, sL1, sL2, sL3):
        super(ExtendedAnomalyNet, self).__init__()
        self.imH = imH
        self.imW = imW

        self.multiPoolPrepare = multiPoolPrepare(pH, pW)
        self.conv1 = base_net.conv1
        self.multiMaxPooling1 = multiMaxPooling(sL1, sL1, sL1, sL1)
        self.conv2 = base_net.conv2
        self.multiMaxPooling2 = multiMaxPooling(sL2, sL2, sL2, sL2)
        self.conv3 = base_net.conv3
        self.multiMaxPooling3 = multiMaxPooling(sL3, sL3, sL3, sL3)
        self.conv4 = base_net.conv4
        self.conv5 = base_net.conv5

        self.outChans = self.conv5.out_channels
        self.unwrapPrepare = unwrapPrepare()
        self.unwrapPool3 = unwrapPool(self.outChans, imH / (sL1 * sL2 * sL3), imW / (sL1 * sL2 * sL3), sL3, sL3)
        self.unwrapPool2 = unwrapPool(self.outChans, imH / (sL1 * sL2), imW / (sL1 * sL2), sL2, sL2)
        self.unwrapPool1 = unwrapPool(self.outChans, imH / sL1, imW / sL1, sL1, sL1)

        self.decode = base_net.decode
        self.decodeChans = self.decode.out_features
        self.l_relu = base_net.l_relu

    def forward(self, x):
        x = self.multiPoolPrepare(x)

        x = self.l_relu(self.conv1(x))
        x = self.multiMaxPooling1(x)

        x = self.l_relu(self.conv2(x))
        x = self.multiMaxPooling2(x)

        x = self.l_relu(self.conv3(x))
        x = self.multiMaxPooling3(x)

        x = self.l_relu(self.conv4(x))
        x = self.l_relu(self.conv5(x))

        x = self.unwrapPrepare(x)
        x = self.unwrapPool3(x)
        x = self.unwrapPool2(x)
        x = self.unwrapPool1(x)

        y = x.view(self.outChans, self.imH, self.imW, -1)
        y = y.permute(3, 1, 2, 0)
        y = self.l_relu(self.decode(y))
        return y


if __name__ == '__main__':
    from AnomalyNet import AnomalyNet
    import numpy as np
    ## Batch size
    batch_size = 2

    ## Image size
    imH = 256
    imW = 256

    ## Stride step size
    sL1 = 2
    sL2 = 2
    sL3 = 2

    ## Define patch dimensions
    pW = 65
    pH = 65

    ## Define test image
    testImage = torch.randn(batch_size, 3, imH, imW)
    testImage = testImage.cuda()

    ## Adjust image dimensions to divide by S x S without remainder
    imW = int(np.ceil(imW / (sL1 * sL2 * sL3)) * sL1 * sL2 * sL3)
    imH = int(np.ceil(imH / (sL1 * sL2 * sL3)) * sL1 * sL2 * sL3)

    ## Base_net definitions
    base_net = AnomalyNet()
    base_net.cuda()
    base_net.eval()
    ## ExtendedAnomalyNet definitions & test run
    slim_net = ExtendedAnomalyNet(base_net=base_net, pH=pH, pW=pW, sL1=sL1, sL2=sL2, sL3=sL3, imH=imH, imW=imW)
    slim_net.cuda()
    slim_net.eval()

    y1 = slim_net(testImage)
    print(y1.size())

    patch = testImage[:, :, :65, :65]
    print(f'patch center: (32, 32), path size: {patch.size()}')

    y2 = base_net(testImage[:, :, :65, :65])
    print(y2.size())
    err =  torch.mean((y1[:, 32, 32, :] - y2)**2).item()
    print(f'error on pixel (32, 32): {err}')

    mean_val_pixel_y1 = torch.mean((y1[:, 32, 32, :]**2)).item()
    print(mean_val_pixel_y1)

    mean_val_pixel_y2 = torch.mean(y2**2).item()
    print(mean_val_pixel_y2)

    rel_err = 2 * err / (mean_val_pixel_y1 + mean_val_pixel_y2)
    print(f'Relative error: {rel_err}')

    summary(slim_net, (3, 256, 256))


