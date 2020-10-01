import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from FDFE import multiPoolPrepare


class ExtendedAnomalyNet(nn.Module):
    '''Apply patch-based AnomalyNet CNN to an entire image'''
    def __init__(self, base_net, pH, pW, imH, imW):
        super(ExtendedAnomalyNet, self).__init__()
        self.imH = imH
        self.imW = imW
        self.pH = pH
        self.pW = pW
        self.base_net = base_net
        self.outChans = self.base_net.decode.out_features
        self.multiPoolPrepare = multiPoolPrepare(pH, pW) # this pads the input image with zeros
    
    def forward(self, x):
        x = self.multiPoolPrepare(x)
        b, _, _, _ = x.size()
        y = torch.zeros(b, self.outChans, self.imH, self.imW)
        for i in range(self.imH):
            for j in range(self.imW):
                y[:, :, i, j] = self.base_net(x[:, :, i:i+self.pH, j:j+self.pW])
        return y


if __name__ == '__main__':
    from AnomalyNet import AnomalyNet
    ## Batch size
    batch_size = 1

    ## Image size
    imH = 256
    imW = 256

    ## Define patch dimensions
    pW = 65
    pH = 65

    ## Define test image
    testImage = torch.randn(batch_size, 3, imH, imW)
    testImage = testImage.cuda()

    ## Base_net definitions
    base_net = AnomalyNet()
    base_net.cuda()
    base_net.eval()

    ## ExtendedAnomalyNet definitions & test run
    extended_net = ExtendedAnomalyNet(base_net=base_net, pH=pH, pW=pW, imH=imH, imW=imW)
    extended_net.cuda()
    extended_net.eval()

    # y1 = extended_net(testImage).cpu()
    # print(y1.size())

    # patch = testImage[:, :, :65, :65]
    # print(f'patch center: (32, 32), path size: {patch.size()}')

    # y2 = base_net(testImage[:, :, :65, :65]).cpu()
    # print(y2.size())
    # err =  torch.mean((y1[:, :, 32, 32] - y2)**2).item()
    # print(f'error on pixel (32, 32): {err}')

    # mean_val_pixel_y1 = torch.mean((y1[:, :, 32, 32]**2)).item()
    # print(mean_val_pixel_y1)

    # mean_val_pixel_y2 = torch.mean(y2**2).item()
    # print(mean_val_pixel_y2)

    # rel_err = 2 * err / (mean_val_pixel_y1 + mean_val_pixel_y2)
    # print(f'Relative error: {rel_err}')

    summary(extended_net, (3, 256, 256))