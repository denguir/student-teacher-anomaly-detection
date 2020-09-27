# Source:
# https://github.com/erezposner/Fast_Dense_Feature_Extraction

from torch import nn
import torch
import numpy as np
import torch.nn.functional as F


# (N,C,H,W)


class multiPoolPrepare(nn.Module):
    def __init__(self, patchY, patchX):
        super(multiPoolPrepare, self).__init__()
        pady = patchY - 1
        padx = patchX - 1

        self.pad_top = np.ceil(pady / 2).astype(int)
        self.pad_bottom = np.floor(pady / 2).astype(int)
        self.pad_left = np.ceil(padx / 2).astype(int)
        self.pad_right = np.floor(padx / 2).astype(int)

    def forward(self, x):
        y = F.pad(x, [self.pad_left, self.pad_right, self.pad_top, self.pad_bottom], value=0)
        return y


class unwrapPrepare(nn.Module):
    def __init__(self):
        super(unwrapPrepare, self).__init__()

    def forward(self, x):
        x_ = F.pad(x, [0, -1, 0, -1], value=0)
        y = x_.contiguous().view(x_.shape[0], -1)
        y = y.transpose(0, 1)
        return y.contiguous()


class unwrapPool(nn.Module):
    def __init__(self, outChans, curImgW, curImgH, dW, dH):
        super(unwrapPool, self).__init__()
        self.outChans = int(outChans)
        self.curImgW = int(curImgW)
        self.curImgH = int(curImgH)
        self.dW = int(dW)
        self.dH = int(dH)

    def forward(self, x, ):
        y = x.view((self.outChans, self.curImgW, self.curImgH, self.dH, self.dW, -1))
        y = y.transpose(2, 3)

        return y.contiguous()


class multiMaxPooling(nn.Module):
    def __init__(self, kW, kH, dW, dH):
        super(multiMaxPooling, self).__init__()
        layers = []
        self.padd = []
        for i in range(0, dH):
            for j in range(0, dW):
                self.padd.append((-j, -i))
                layers.append(nn.MaxPool2d(kernel_size=(kW, kH), stride=(dW, dH)))
        self.max_layers = nn.ModuleList(layers)
        self.s = dH

    def forward(self, x):

        hh = []
        ww = []
        res = []

        for i in range(0, len(self.max_layers)):
            pad_left, pad_top = self.padd[i]
            _x = F.pad(x, [pad_left, pad_left, pad_top, pad_top], value=0)
            _x = self.max_layers[i](_x)
            h, w = _x.size()[2], _x.size()[3]
            hh.append(h)
            ww.append(w)
            res.append(_x)
        max_h, max_w = np.max(hh), np.max(ww)
        for i in range(0, len(self.max_layers)):
            _x = res[i]
            h, w = _x.size()[2], _x.size()[3]
            pad_top = np.floor((max_h - h) / 2).astype(int)
            pad_bottom = np.ceil((max_h - h) / 2).astype(int)
            pad_left = np.floor((max_w - w) / 2).astype(int)
            pad_right = np.ceil((max_w - w) / 2).astype(int)
            _x = F.pad(_x, [pad_left, pad_right, pad_top, pad_bottom], value=0)
            res[i] = _x
        return torch.cat(res, 0)



class multiConv(nn.Module):
    def __init__(self, nInputPlane, nOutputPlane, kW, kH, dW, dH):
        super(multiConv, self).__init__()
        layers = []
        self.padd = []
        for i in range(0, dH):
            for j in range(0, dW):
                self.padd.append((-j, -i))
                torch.manual_seed(10)
                torch.cuda.manual_seed(10)
                a = nn.Conv2d(nInputPlane, nOutputPlane, kernel_size=(kW, kH), stride=(dW, dH), padding=0)
                layers.append(a)
        self.max_layers = nn.ModuleList(layers)
        self.s = dW

    def forward(self, x):
        hh = []
        ww = []
        res = []

        for i in range(0, len(self.max_layers)):
            pad_left, pad_top = self.padd[i]
            _x = F.pad(x, [pad_left, pad_left, pad_top, pad_top], value=0)
            _x = self.max_layers[i](_x)
            h, w = _x.size()[2], _x.size()[3]
            hh.append(h)
            ww.append(w)
            res.append(_x)
        max_h, max_w = np.max(hh), np.max(ww)
        for i in range(0, len(self.max_layers)):
            _x = res[i]
            h, w = _x.size()[2], _x.size()[3]
            pad_top = np.ceil((max_h - h) / 2).astype(int)
            pad_bottom = np.floor((max_h - h) / 2).astype(int)
            pad_left = np.ceil((max_w - w) / 2).astype(int)
            pad_right = np.floor((max_w - w) / 2).astype(int)
            _x = F.pad(_x, [pad_left, pad_right, pad_top, pad_bottom], value=0)
            res[i] = _x
        return torch.cat(res, 0)
