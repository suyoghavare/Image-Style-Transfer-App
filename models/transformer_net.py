import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = in_channels

        self.conv1 = nn.Conv2d(
            in_channels, mid_channels,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.in1 = nn.InstanceNorm2d(mid_channels, affine=True)

        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels,
            kernel_size=3, stride=1, padding=1,
            groups=mid_channels, bias=False
        )
        self.in2 = nn.InstanceNorm2d(mid_channels, affine=True)

        self.conv3 = nn.Conv2d(
            mid_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.in3 = nn.InstanceNorm2d(out_channels, affine=True)

        self.relu = nn.ReLU(inplace=True)
        self.use_residual = (in_channels == out_channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.relu(self.in2(self.conv2(out)))
        out = self.in3(self.conv3(out))
        if self.use_residual:
            out = out + identity
        out = self.relu(out)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.conv1 = DepthwiseSeparableBlock(in_channels, out_channels)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.conv1(x)
        return x


class TransformerNet(nn.Module):
    """
    Architecture for mosaic.pth / picasso.pth / candy.pth
    """
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

        return y  # scaling handled in utils/style_transfer.py

# --------------------------------------------------------------------------------------
# Johnson-style network for udnie / wave / starry / lazy / tokyo_ghoul
# (keys like "ConvBlock.*", "ResidualBlock.*", "DeconvBlock.*")
# --------------------------------------------------------------------------------------

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
        y = self.relu(y)
        return y


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
        y = self.relu(y)
        return y


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
    """
    Architecture for udnie.pth, wave.pth, starry.pth, lazy.pth, tokyo_ghoul.pth
    (classic Johnson-style fast style transfer network)
    """
    def __init__(self):
        super().__init__()

        # indices 0/2/4 are used in the checkpoint as ConvBlock.0, .2, .4
        self.ConvBlock = nn.ModuleList([
            ConvBlockModule(3, 32, 9, 1),   # ConvBlock.0
            nn.Identity(),                  # ConvBlock.1 (no params)
            ConvBlockModule(32, 64, 3, 2),  # ConvBlock.2
            nn.Identity(),                  # ConvBlock.3
            ConvBlockModule(64, 128, 3, 2), # ConvBlock.4
        ])

        self.ResidualBlock = nn.ModuleList([
            ResidualModule(128),  # ResidualBlock.0
            ResidualModule(128),  # ResidualBlock.1
            ResidualModule(128),  # ResidualBlock.2
            ResidualModule(128),  # ResidualBlock.3
            ResidualModule(128),  # ResidualBlock.4
        ])

        # indices 0/2/4 used as DeconvBlock.0, .2, .4
        self.DeconvBlock = nn.ModuleList([
            DeconvUpModule(128, 64),   # DeconvBlock.0
            nn.Identity(),             # DeconvBlock.1
            DeconvUpModule(64, 32),    # DeconvBlock.2
            nn.Identity(),             # DeconvBlock.3
            DeconvOutModule(32, 3),    # DeconvBlock.4
        ])

    def forward(self, x):
        # encoder
        x = self.ConvBlock[0](x)
        x = self.ConvBlock[2](x)
        x = self.ConvBlock[4](x)

        # residual blocks
        for res in self.ResidualBlock:
            x = res(x)

        # decoder
        x = self.DeconvBlock[0](x)
        x = self.DeconvBlock[2](x)
        x = self.DeconvBlock[4](x)

        return x
