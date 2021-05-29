import torch.nn as nn
import torch.nn.functional as F

class Hsigmoid(nn.Module):
    """
    Hard sigmoid function
    """
    def __init__(self, inplace: bool = True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.relu6(input + 3.0, inplace=self.inplace) * (1.0/6.0)


class Hswish(nn.Module):
    """
    Hard swish function
    """
    def __init__(self, inplace: bool = True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.relu6(input+3.0, inplace=self.inplace) * (1.0/6.0) * input


class Squeeze_excite(nn.Module):
    def __init__(self, num_channels, r=4):
        """
        Squeeze-and-Excitation block
          Args:
            num_channels (int): number of channels in the input tensor
            r (int): num_channels are divided by r in the first conv block
        """
        super(Squeeze_excite, self).__init__()

        #instead of fully connected layers 1x1 convolutions are used, which has exactly the same effect as the input tensor is 1x1 after pooling
        #batch normalization is not used here as it's absent in the paper
        self.conv_0 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_channels, num_channels//r, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(num_channels//r, num_channels, kernel_size=1),
            Hsigmoid()
        )

    def forward(self, input):
        out = self.conv_0(input)
        out = self.conv_1(out)
        out = out * input
        return out


class Bneck(nn.Module):
    def __init__(self, nin, nout, k_size, exp_size, se, act, s, wm=1.0):
        """
        Bottleneck block
          Args:
            nin (int): number of channels in the input tensor
            nout (int): number of channels in the output tensor
            k_size (int): size of filters
            exp_size (int): expansion size
            se (bool): whether to use Squeeze-and-Excitation
            act (nn.Module): activation function
            s (int): stride
            wm (float): width multiplier
        """
        super(Bneck, self).__init__()
        nin = int(nin*wm)
        nout = int(nout*wm)
        exp_size = int(exp_size*wm)

        self.pointwise_0 = nn.Sequential(
            nn.Conv2d(nin, exp_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(exp_size),
            act(inplace=True)
        )
        
        self.depthwise_1 = nn.Sequential(
            nn.Conv2d(exp_size, exp_size, kernel_size=k_size, padding=(k_size-1)//2, groups=exp_size, stride=s, bias=False),
            nn.BatchNorm2d(exp_size),
            act(inplace=True)
        )

        self.se = se
        self.se_block = Squeeze_excite(num_channels=exp_size, r=4)
        
        self.pointwise_2 = nn.Sequential(
            nn.Conv2d(exp_size, nout, kernel_size=1, bias=False),
            nn.BatchNorm2d(nout)
        )
        
        self.shortcut = s == 1 and nin == nout

    def forward(self, input):
        identity = input
        out = self.pointwise_0(input)
        out = self.depthwise_1(out)
        
        if self.se:
            out = self.se_block(out)
        
        out = self.pointwise_2(out)
        
        if self.shortcut:
            out += identity
        return out


class Mobilenet_v3_large(nn.Module):
    def __init__(self, wm=1.0, si=1, drop_prob=0.0):
        """
        Mobilenet v3 large model
          Args:
            wm (float): width multiplier
            si (int): stride in initial layers (set to 1 by default instead of 2 to adapt for small 32x32 resolution of CIFAR)
            drop_prob (float): probability that a neuron is removed for nn.Dropout layer during training
        """
        super(Mobilenet_v3_large, self).__init__()
        self.wm = wm
        
        self.conv_0 = nn.Sequential(
            nn.Conv2d(3, int(16*wm), 3, padding=1, stride=si, bias=False),
            nn.BatchNorm2d(int(16*wm)),
            Hswish()
        )
        
        self.bottlenecks_1 = nn.Sequential(
            Bneck(nin=16,  nout=16,  k_size=3, exp_size=16,  se=False, act=nn.ReLU, s=1,  wm=wm),#1
            Bneck(nin=16,  nout=24,  k_size=3, exp_size=64,  se=False, act=nn.ReLU, s=si, wm=wm),#2
            Bneck(nin=24,  nout=24,  k_size=3, exp_size=72,  se=False, act=nn.ReLU, s=1,  wm=wm),#3
            Bneck(nin=24,  nout=40,  k_size=5, exp_size=72,  se=True,  act=nn.ReLU, s=si, wm=wm),#4
            Bneck(nin=40,  nout=40,  k_size=5, exp_size=120, se=True,  act=nn.ReLU, s=1,  wm=wm),#5
            Bneck(nin=40,  nout=40,  k_size=5, exp_size=120, se=True,  act=nn.ReLU, s=1,  wm=wm),#6
            Bneck(nin=40,  nout=80,  k_size=3, exp_size=240, se=False, act=Hswish,  s=2,  wm=wm),#7
            Bneck(nin=80,  nout=80,  k_size=3, exp_size=200, se=False, act=Hswish,  s=1,  wm=wm),#8
            Bneck(nin=80,  nout=80,  k_size=3, exp_size=184, se=False, act=Hswish,  s=1,  wm=wm),#9
            Bneck(nin=80,  nout=80,  k_size=3, exp_size=184, se=False, act=Hswish,  s=1,  wm=wm),#10
            Bneck(nin=80,  nout=112, k_size=3, exp_size=480, se=True,  act=Hswish,  s=1,  wm=wm),#11
            Bneck(nin=112, nout=112, k_size=3, exp_size=672, se=True,  act=Hswish,  s=1,  wm=wm),#12
            Bneck(nin=112, nout=160, k_size=5, exp_size=672, se=True,  act=Hswish,  s=2,  wm=wm),#13
            Bneck(nin=160, nout=160, k_size=5, exp_size=960, se=True,  act=Hswish,  s=1,  wm=wm),#14
            Bneck(nin=160, nout=160, k_size=5, exp_size=960, se=True,  act=Hswish,  s=1,  wm=wm) #15
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(int(160*wm), int(960*wm), 1, bias=False),
            nn.BatchNorm2d(int(960*wm)),
            Hswish()
        )
        
        self.conv_3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(int(960*wm), 1280, 1),
            Hswish(),
            nn.Dropout(p=drop_prob)
        )
        
        self.conv_4 = nn.Conv2d(1280, 100, 1)

    def forward(self, input):
        x = self.conv_0(input)
        x = self.bottlenecks_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        return x.view(x.shape[0], -1)
    
    def name(self):
        return "Mobilenet_v3_large_" + str(self.wm)


class Mobilenet_v3_small(nn.Module):
    def __init__(self, wm=1.0, si=1, drop_prob=0.0):
        """
        Mobilenet v3 small model
          Args:
            wm (float): width multiplier
            si (int): stride in initial layers (set to 1 by default instead of 2 to adapt for small 32x32 resolution of CIFAR)
            drop_prob (float): probability that a neuron is removed for nn.Dropout layer
        """
        super(Mobilenet_v3_small, self).__init__()
        self.wm = wm
        
        self.conv_0 = nn.Sequential(
            nn.Conv2d(3, int(16*wm), 3, padding=1, stride=si, bias=False),
            nn.BatchNorm2d(int(16*wm)),
            Hswish()
        )
        
        self.bottlenecks_1 = nn.Sequential(
            Bneck(nin=16,  nout=16,  k_size=3, exp_size=16,  se=True,  act=nn.ReLU, s=si, wm=wm),#1 
            Bneck(nin=16,  nout=24,  k_size=3, exp_size=72,  se=False, act=nn.ReLU, s=si, wm=wm),#2
            Bneck(nin=24,  nout=24,  k_size=3, exp_size=88,  se=False, act=nn.ReLU, s=1,  wm=wm),#3
            Bneck(nin=24,  nout=40,  k_size=5, exp_size=96,  se=True,  act=Hswish,  s=2,  wm=wm),#4
            Bneck(nin=40,  nout=40,  k_size=5, exp_size=240, se=True,  act=Hswish,  s=1,  wm=wm),#5
            Bneck(nin=40,  nout=40,  k_size=5, exp_size=240, se=True,  act=Hswish,  s=1,  wm=wm),#6
            Bneck(nin=40,  nout=48,  k_size=5, exp_size=120, se=True,  act=Hswish,  s=1,  wm=wm),#7
            Bneck(nin=48,  nout=48,  k_size=5, exp_size=144, se=True,  act=Hswish,  s=1,  wm=wm),#8
            Bneck(nin=48,  nout=96,  k_size=5, exp_size=288, se=True,  act=Hswish,  s=2,  wm=wm),#9
            Bneck(nin=96,  nout=96,  k_size=5, exp_size=576, se=True,  act=Hswish,  s=1,  wm=wm),#10
            Bneck(nin=96,  nout=96,  k_size=5, exp_size=576, se=True,  act=Hswish,  s=1,  wm=wm) #11
        )

        #there is an SE block after Hswish according to the paper, however it seems to be a mistake
        #SE block is not present in the oficial tensorflow code [https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v3.py]
        self.conv_2 = nn.Sequential(
            nn.Conv2d(int(96*wm), int(576*wm), 1, bias=False),
            nn.BatchNorm2d(int(576*wm)),
            Hswish()
        )
        
        self.conv_3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(int(576*wm), 1024, 1),
            Hswish(),
            nn.Dropout(p=drop_prob)
        )
        
        self.conv_4 = nn.Conv2d(1024, 100, 1)


    def forward(self, input):
        x = self.conv_0(input)
        x = self.bottlenecks_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        return x.view(x.shape[0], -1)
    
    def name(self):
        return "Mobilenet_v3_small_" + str(self.wm)
