import torch
import torch.nn.functional as F

from unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, dropout_rate=0.3):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64, dropout_rate=dropout_rate)
        self.down1 = Down(64, 128, dropout_rate=dropout_rate)
        self.down2 = Down(128, 256, dropout_rate=dropout_rate)
        self.down3 = Down(256, 512, dropout_rate=dropout_rate)
        self.down4 = Down(512, 512, dropout_rate=dropout_rate)
        self.up1 = Up(1024, 256, bilinear, dropout_rate=dropout_rate)
        self.up2 = Up(512, 128, bilinear, dropout_rate=dropout_rate)
        self.up3 = Up(256, 64, bilinear, dropout_rate=dropout_rate)
        self.up4 = Up(128, 64, bilinear, dropout_rate=dropout_rate)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    net = UNet(n_channels=3, n_classes=1)
    print(net)