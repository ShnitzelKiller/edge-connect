import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .ops import contextual_attention, contextual_patches_score, contextual_patches_reconstruction, contextual_scores, contextual_reconstruction
from .util import *


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, K=3, S=1, P=1, D=1, activation=nn.ELU()):
        super(Conv, self).__init__()
        if activation:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=K, stride=S, padding=P, dilation=D),
                activation
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=K, stride=S, padding=P, dilation=D)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


# conv 1~6
class Down_Module(nn.Module):
    def __init__(self, in_ch, out_ch, activation=nn.ELU()):
        super(Down_Module, self).__init__()
        layers = []
        layers.append(Conv(in_ch, out_ch, K=5))

        curr_dim = out_ch
        for i in range(2):
            layers.append(Conv(curr_dim, curr_dim*2, K=3, S=2))
            layers.append(Conv(curr_dim*2, curr_dim*2))
            curr_dim *= 2
        
        layers.append(Conv(curr_dim, curr_dim, activation=activation))
        self.out = nn.Sequential(*layers)

    def forward(self, x):
        return self.out(x)


# conv 7~10
class Dilation_Module(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Dilation_Module, self).__init__()
        layers = []
        dilation = 1
        for i in range(4):
            dilation *= 2
            layers.append(Conv(in_ch, out_ch, D=dilation, P=dilation))
        self.out = nn.Sequential(*layers)

    def forward(self, x):
        return self.out(x)


# conv 11~17
class Up_Module(nn.Module):
    def __init__(self, in_ch, out_ch, isRefine=False):
        super(Up_Module, self).__init__()
        layers = []
        curr_dim = in_ch
        if isRefine:
            layers.append(Conv(curr_dim, curr_dim//2))
            curr_dim //= 2
        else:
            layers.append(Conv(curr_dim, curr_dim))
        
        # conv 12~15
        for i in range(2):
            layers.append(Conv(curr_dim, curr_dim))
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(Conv(curr_dim, curr_dim//2))
            curr_dim //= 2

        layers.append(Conv(curr_dim, curr_dim//2))
        layers.append(Conv(curr_dim//2, out_ch, activation=0))

        self.out = nn.Sequential(*layers)

    def forward(self, x):
        output = self.out(x)
        return torch.clamp(output, min=-1., max=1.)


class Flatten_Module(nn.Module):
    def __init__(self, in_ch, out_ch, isLocal=True):
        super(Flatten_Module, self).__init__()
        layers = []
        layers.append(Conv(in_ch, out_ch, K=5, S=2, P=2, activation=nn.LeakyReLU()))
        curr_dim = out_ch

        for i in range(2):
            layers.append(Conv(curr_dim, curr_dim*2, K=5, S=2, P=2, activation=nn.LeakyReLU()))
            curr_dim *= 2
        
        if isLocal:
            layers.append(Conv(curr_dim, curr_dim*2, K=5, S=2, P=2, activation=nn.LeakyReLU()))
        else:
            layers.append(Conv(curr_dim, curr_dim, K=5, S=2, P=2, activation=nn.LeakyReLU()))

        self.out = nn.Sequential(*layers)

    def forward(self, x):
        x = self.out(x)
        return x.view(x.size(0),-1) # 2B x 256*(256 or 512); front 256:16*16


# pmconv 9~10
class Contextual_Attention_Module(nn.Module):
    def __init__(self, in_ch, out_ch, rate=2, stride=1, ksize=3, fuse_k=3, softmax_scale=10., fuse=True):
        super(Contextual_Attention_Module, self).__init__()
        self.rate = rate
        self.stride = stride
        self.ksize = ksize
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        #self.padding = nn.ZeroPad2d(1)
        layers = []
        for i in range(2):
            layers.append(Conv(in_ch, out_ch).cuda())
        self.out = nn.Sequential(*layers)

    def forward(self, f, b, mask=None, visualize=False):
        
        if visualize:
            y, flow = contextual_attention(f, b, mask=mask, ksize=self.ksize, stride=self.stride, rate=self.rate, fuse_k=self.fuse_k, softmax_scale=self.softmax_scale, fuse=self.fuse, visualize=True)
            return self.out(y), flow
        else:
            y = contextual_attention(f, b, mask=mask, ksize=self.ksize, stride=self.stride, rate=self.rate, fuse_k=self.fuse_k, softmax_scale=self.softmax_scale, fuse=self.fuse, visualize=False)
            return self.out(y)

class Contextual_Patches_Score_Module(nn.Module):
    def __init__(self, ksize=3, stride=1, rate=2):
        super(Contextual_Patches_Score_Module, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
    def forward(self, f, b):
        return contextual_patches_score(f, b, ksize=self.ksize, stride=self.stride, rate=self.rate)


class Contextual_Patches_Reconstruction_Module(nn.Module):
    def __init__(self, ksize=3, stride=1, rate=2):
        super(Contextual_Patches_Reconstruction_Module, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
    def forward(self, b, mask=None):
        return contextual_patches_reconstruction(b, mask=mask, ksize=self.ksize, stride=self.stride, rate=self.rate)


class Contextual_Score_Module(nn.Module):
    def __init__(self, ksize=3):
        super(Contextual_Score_Module, self).__init__()
        self.ksize=ksize
    def forward(self, f, w):
        return contextual_scores(f, w, ksize=self.ksize)


class Contextual_Reconstruction_Module(nn.Module):
    def __init__(self, in_ch, out_ch, rate=2, fuse_k=3, softmax_scale=10., fuse=True):
        super(Contextual_Reconstruction_Module, self).__init__()
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        layers = []
        for i in range(2):
            layers.append(Conv(in_ch, out_ch).cuda())
        self.out = nn.Sequential(*layers)
    def forward(self, raw_w, mm, scores, raw_int_fs, int_fs, int_bs, visualize=False):
        if visualize:
            y, flow = contextual_reconstruction(raw_w, mm, scores, raw_int_fs, int_fs, int_bs, rate=self.rate, fuse_k=self.fuse_k, softmax_scale=self.softmax_scale, fuse=self.fuse, visualize=True)
            return self.out(y), flow
        else:
            y = contextual_reconstruction(raw_w, mm, scores, raw_int_fs, int_fs, int_bs, rate=self.rate, fuse_k=self.fuse_k, softmax_scale=self.softmax_scale, fuse=self.fuse, visualize=False)
            return self.out(y)


def test_contextual_attention(args):
    """Test contextual attention layer with 3-channel image input
    (instead of n-channel feature).

    """
    import cv2
    import os
    import matplotlib.pyplot as plt
    rate = 2
    stride = 1
    grid = rate*stride

    b = cv2.imread(args.imageA)
    h, w, _ = b.shape
    b = b[:h//grid*grid, :w//grid*grid, :]
    b = np.transpose(b,[2,0,1])
    b = np.expand_dims(b, 0)
    print('Size of imageA: {}'.format(b.shape))

    f = cv2.imread(args.imageB)
    h, w, _ = f.shape
    f = f[:h//grid*grid, :w//grid*grid, :]
    f = np.transpose(f,[2,0,1])
    f = np.expand_dims(f, 0)
    print('Size of imageB: {}'.format(f.shape))

    bt = torch.Tensor(b).cuda()
    #plt.imshow(np.transpose(np.clip(bt[0,:,:,:].cpu().data.numpy(),0,255).astype(np.uint8), [1, 2, 0]))
    #plt.show()
    ft = torch.Tensor(f).cuda()
    #plt.imshow(np.transpose(np.clip(ft[0,:,:,:].cpu().data.numpy(),0,255).astype(np.uint8), [1, 2, 0]))
    #plt.show()
    print('ftdevice:',ft.device)
    print('btdevice:',bt.device)
    yt, flow = contextual_attention(
        ft, bt, stride=stride,
        training=False, fuse=False)
    y = yt.cpu().data.numpy().transpose([0,2,3,1])
    #plt.imshow(flow[0].cpu().data.numpy().transpose([1,2,0]))
    #plt.show()
    outImg = np.clip(y[0],0,255).astype(np.uint8)
    plt.imshow(outImg)
    plt.show()
    cv2.imwrite(args.imageOut, outImg)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--imageA', default='', type=str, help='Image A as background patches to reconstruct image B.')
    parser.add_argument('--imageB', default='', type=str, help='Image B is reconstructed with image A.')
    parser.add_argument('--imageOut', default='result.png', type=str, help='Image B is reconstructed with image A.')
    args = parser.parse_args()
    test_contextual_attention(args)
