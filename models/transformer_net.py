import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = in_channels

        self.conv1 = nn.Conv2d(
            in_channels, mid,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.in1 = nn.InstanceNorm2d(mid, affine=True)

        self.conv2 = nn.Conv2d(
            mid, mid,
            kernel_size=3, stride=1, padding=1,
            groups=mid, bias=False
        )
        self.in2 = nn.InstanceNorm2d(mid, affine=True)

        self.conv3 = nn.Conv2d(
            mid, out_channels,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.in3 = nn.InstanceNorm2d(out_channels, affine=True)

        self.relu = nn.ReLU(inplace=True)
        self.use_residual = (in_channels == out_channels)

    def forward(self, x):
        identity = x

        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.in3(self.conv3(y))

        if self.use_residual:
            y = y + identity

        return self.relu(y)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.conv1 = DepthwiseSeparableBlock(in_channels, out_channels)
        self.scale_factor = scale_factor

    def forward(self, x):
        y = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        return self.conv1(y)


class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3, 32, kernel_size=9, stride=1, padding=4, bias=False
        )
        self.in1 = nn.InstanceNorm2d(32, affine=True)

        self.conv2 = DepthwiseSeparableBlock(32, 64)
        self.conv3 = DepthwiseSeparableBlock(64, 128)

        self.res1 = DepthwiseSeparableBlock(128, 128)
        self.res2 = DepthwiseSeparableBlock(128, 128)
        self.res3 = DepthwiseSeparableBlock(128, 128)
        self.res4 = DepthwiseSeparableBlock(128, 128)
        self.res5 = DepthwiseSeparableBlock(128, 128)

        self.upconv1 = UpsampleBlock(128, 64)
        self.upconv2 = UpsampleBlock(64, 32)

        self.conv4 = nn.Conv2d(
            32, 3, kernel_size=9, stride=1, padding=4, bias=False
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.in1(self.conv1(x)))
        y = self.conv2(y)
        y = self.conv3(y)

        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)

        y = self.upconv1(y)
        y = self.upconv2(y)
        y = self.conv4(y)

        return y


#Johnson-style network

class ConvBlockModule(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride):
        super().__init__()
        padding = kernel_size // 2
        self.conv_layer = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.norm_layer = nn.InstanceNorm2d(out_c, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.conv_layer(x)
        y = self.norm_layer(y)
        return self.relu(y)


class ResidualModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlockModule(channels, channels, 3, 1)
        self.conv2 = ConvBlockModule(channels, channels, 3, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return x + y


class DeconvUpModule(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_c, out_c,
            kernel_size=3, stride=2,
            padding=1, output_padding=1
        )
        self.norm_layer = nn.InstanceNorm2d(out_c, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.conv_transpose(x)
        y = self.norm_layer(y)
        return self.relu(y)


class DeconvOutModule(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv_layer = nn.Conv2d(
            in_c, out_c,
            kernel_size=9, stride=1, padding=4
        )

    def forward(self, x):
        return self.conv_layer(x)


class JohnsonNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.ConvBlock = nn.ModuleList([
            ConvBlockModule(3, 32, 9, 1),    
            nn.Identity(),                   
            ConvBlockModule(32, 64, 3, 2),    
            nn.Identity(),                   
            ConvBlockModule(64, 128, 3, 2),  
        ])

        self.ResidualBlock = nn.ModuleList([
            ResidualModule(128),  
            ResidualModule(128),  
            ResidualModule(128), 
            ResidualModule(128), 
            ResidualModule(128), 
        ])

        self.DeconvBlock = nn.ModuleList([
            DeconvUpModule(128, 64),   
            nn.Identity(),             
            DeconvUpModule(64, 32),    
            nn.Identity(),             
            DeconvOutModule(32, 3),    
        ])

    def forward(self, x):
        x = self.ConvBlock[0](x)
        x = self.ConvBlock[2](x)
        x = self.ConvBlock[4](x)

        for res in self.ResidualBlock:
            x = res(x)

        x = self.DeconvBlock[0](x)
        x = self.DeconvBlock[2](x)
        x = self.DeconvBlock[4](x)

        return x
