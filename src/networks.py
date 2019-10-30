import torch
import torch.nn as nn
from .model_module import Contextual_Attention_Module, Contextual_Patches_Reconstruction_Module, Contextual_Patches_Score_Module, Contextual_Score_Module, Contextual_Reconstruction_Module
import torch.nn.functional as F

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class InpaintGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True, contextual_attention=False, multi_contextual_attention=False, visualize_contextual_attention=False, use_objmasks=False, ksize=3, skip_connections=False):
        super(InpaintGenerator, self).__init__()
        if contextual_attention:
            print('using contextual attention')
        else:
            print('not using contextual attention in inpaint generator')
        if use_objmasks:
            print('using objectmasks in inpaint generator')
        else:
            print('not using objmasks in inpaint generator')
        if skip_connections:
            print('using skip connections in inpaint generator')
            self.base_channels=32
        else:
            print('not using skip connections in inpaint generator')
            self.base_channels=64
        self.use_contextual_attention = contextual_attention
        self.use_multi_contextual_attention = multi_contextual_attention
        self.skip_connections = skip_connections
    
        if self.use_multi_contextual_attention:
            print('using multi contextual attention in inpaint generator')
        else:
            print('not using multi contextual attention in inpaint generator')
        
        if self.skip_connections:
            self.encoder_conv1 = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels=(5 if use_objmasks else 4), out_channels=self.base_channels, kernel_size=7, padding=0),
                nn.ReLU(True),
                nn.BatchNorm2d(self.base_channels, track_running_stats=False),
                
            )

            self.encoder_conv2 = nn.Sequential(
                nn.Conv2d(in_channels=self.base_channels, out_channels=self.base_channels*2, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(self.base_channels*2, track_running_stats=False),
                nn.Conv2d(in_channels=self.base_channels*2, out_channels=self.base_channels*2, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(self.base_channels*2, track_running_stats=False),
            )

            self.maxpool1 = nn.MaxPool2d(2)

            self.encoder_conv3 = nn.Sequential(
                nn.Conv2d(in_channels=self.base_channels*2, out_channels=self.base_channels*4, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(self.base_channels*4, track_running_stats=False),
                nn.Conv2d(in_channels=self.base_channels*4, out_channels=self.base_channels*4, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(self.base_channels*4, track_running_stats=False),
            )
            
            self.maxpool2 = nn.MaxPool2d(2)
            
        else:
            if self.use_multi_contextual_attention:
                self.encoder1 = nn.Sequential(
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(in_channels=(5 if use_objmasks else 4), out_channels=64, kernel_size=7, padding=0),
                    nn.InstanceNorm2d(64, track_running_stats=False),
                    nn.ReLU(True),
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(128, track_running_stats=False),
                    nn.ReLU(True),
                )
                self.encoder2 = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(256, track_running_stats=False),
                    nn.ReLU(True)
                )
            else:
                self.encoder = nn.Sequential(
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(in_channels=(5 if use_objmasks else 4), out_channels=64, kernel_size=7, padding=0),
                    nn.InstanceNorm2d(64, track_running_stats=False),
                    nn.ReLU(True),
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(128, track_running_stats=False),
                    nn.ReLU(True),
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(256, track_running_stats=False),
                    nn.ReLU(True)
                )
        if self.use_contextual_attention:
            #self.maskpool1 = nn.MaxPool2d(2)
            #self.maskpool2 = nn.MaxPool2d(2)
            if self.use_multi_contextual_attention:
                self.patches_score1 = Contextual_Patches_Score_Module(ksize=ksize, stride=1, rate=2)
                self.patches_recon1 = Contextual_Patches_Reconstruction_Module(ksize=ksize, stride=1, rate=2)
                self.score1 = Contextual_Score_Module(ksize=ksize)
                self.recon1 = Contextual_Reconstruction_Module(self.base_channels*4, self.base_channels*4, rate=2)
                self.patches_recon2 = Contextual_Patches_Reconstruction_Module(ksize=ksize, stride=2, rate=2)
                self.recon2 = Contextual_Reconstruction_Module(self.base_channels*2, self.base_channels*2, rate=2)
            else:
                self.contextual_attention = Contextual_Attention_Module(self.base_channels*4, self.base_channels*4, rate=2, stride=1, ksize=ksize)
        self.visualize_contextual_attention = visualize_contextual_attention
        
        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(self.base_channels*4, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        
        if self.skip_connections:
            self.decoder_conv3 = nn.Sequential(
                nn.Conv2d(in_channels=self.base_channels*4, out_channels=self.base_channels*4, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(self.base_channels*4, track_running_stats=False),
                nn.Conv2d(in_channels=self.base_channels*4, out_channels=self.base_channels*4, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(self.base_channels*4, track_running_stats=False),
                nn.ConvTranspose2d(in_channels=self.base_channels*4, out_channels=self.base_channels*4, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(self.base_channels*4, track_running_stats=False),
            )

            self.decoder_conv2 = nn.Sequential(
                nn.Conv2d(in_channels=self.base_channels*8, out_channels=self.base_channels*4, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(self.base_channels*4, track_running_stats=False),
                nn.Conv2d(in_channels=self.base_channels*4, out_channels=self.base_channels*4, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(self.base_channels*4, track_running_stats=False),
                nn.ConvTranspose2d(self.base_channels*4, out_channels=self.base_channels*2, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(self.base_channels*2, track_running_stats=False),
            )

            self.decoder_conv1 = nn.Sequential(
                nn.Conv2d(in_channels=self.base_channels*4, out_channels=self.base_channels*2, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(self.base_channels*2, track_running_stats=False),
                nn.Conv2d(in_channels=self.base_channels*2, out_channels=self.base_channels*2, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(self.base_channels*2, track_running_stats=False),
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels=self.base_channels*2, out_channels=3, kernel_size=7, padding=0)
            )
        else:
            if self.use_multi_contextual_attention:
                self.decoder2 = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(128, track_running_stats=False),
                    nn.ReLU(True),
                )
                self.decoder1 = nn.Sequential(
                    nn.ConvTranspose2d(256, out_channels=64, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(64, track_running_stats=False),
                    nn.ReLU(True),
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(64, out_channels=3, kernel_size=7, padding=0)
                )
            else:
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(128, track_running_stats=False),
                    nn.ReLU(True),
                    nn.ConvTranspose2d(128, out_channels=64, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(64, track_running_stats=False),
                    nn.ReLU(True),
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(64, out_channels=3, kernel_size=7, padding=0)
                )

        if init_weights:
            self.init_weights()

    def forward(self, x, masks):
        if self.skip_connections:
            x1 = self.encoder_conv1(x)
            x2 = self.encoder_conv2(x1)
            x3 = self.encoder_conv3(self.maxpool1(x2))
            x = self.maxpool2(x3)
        else:
            if self.use_multi_contextual_attention:
                
                x1 = self.encoder1(x)
                x = self.encoder2(x1)
            else:
                x = self.encoder(x)
        if self.use_contextual_attention:
            #masks_s2 = self.maskpool1(masks)
            #masks_s4 = self.maskpool2(masks_s2)
            masks_s2 = F.interpolate(masks, scale_factor=0.5, mode='nearest')
            masks_s4 = F.interpolate(masks, scale_factor=0.25, mode='nearest')
            if self.use_multi_contextual_attention:
                raw_int_fs = list(x.shape)
                f, _, w = self.patches_score1(x, x)
                int_fs = list(f.shape)
                int_bs = list(f.shape)
                raw_w, mm = self.patches_recon1(x, masks_s4)
                scores = self.score1(f, w)
                if self.visualize_contextual_attention:
                    x,flow = self.recon1(raw_w, mm, scores, raw_int_fs, int_fs, int_bs, visualize=True)
                    self.flow = flow
                else:
                    x = self.recon1(raw_w, mm, scores, raw_int_fs, int_fs, int_bs, visualize=False)
            else:
                if self.visualize_contextual_attention:
                    x,flow = self.contextual_attention(x, x, mask=masks_s4, visualize=True)
                    self.flow = flow
                else:
                    x = self.contextual_attention(x, x, mask=masks_s4, visualize=False)
        x = self.middle(x)
        if self.skip_connections:
            x = self.decoder_conv3(x)
            x = self.decoder_conv2(torch.cat((x, x3), 1))
            x = self.decoder_conv1(torch.cat((x, x2), 1))
        else:
            if self.use_multi_contextual_attention:
                scores = F.interpolate(scores, scale_factor=2, mode='bilinear', align_corners=True)
                raw_w, mm = self.patches_recon2(x1, masks_s2)
                int_fs = (x1.shape[0], x1.shape[1], x1.shape[2]//2, x1.shape[3]//2)
                xr = self.recon2(raw_w, mm, scores, list(x1.shape), int_fs, int_bs, visualize=False)
                x = self.decoder2(x)
                x = self.decoder1(torch.cat((x, xr), 1))
            else:
                x = self.decoder(x)
        x = (torch.tanh(x) + 1) / 2

        return x


class EdgeGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, use_spectral_norm=True, init_weights=True, use_objmask=False):
        super(EdgeGenerator, self).__init__()
        if use_objmask:
            print('using objectmasks in edgeGenerator')
        else:
            print('not using objmasks in edgeGenerator')
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=(4 if use_objmask else 3), out_channels=64, kernel_size=7, padding=0), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
