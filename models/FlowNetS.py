import torch
import torch.functional as F
import torch.nn as nn
import numpy as np

class FlowNetS(torch.nn.Module):
    
    def __init__(self):
        super(FlowNetS, self).__init__()

        #256, 256
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=7, stride=2, padding=3)
        # X1 - 128,128
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
        # X2 -64, 64
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)

        self.conv3_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # X3 - 32, 32
        
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool4_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # X4 - 16, 16
        
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.pool5_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # X5 - 8, 8
        
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1)
        # X6 - 8, 8

        # Decoder
        
        self.deconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)
        # X7 - 16, 16

        self.deconv2 = nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=4, stride=2, padding=1)
        # X8 - 32, 32

        self.deconv3 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=4, stride=2, padding=1)
        # X9 - 64, 64

        self.deconv4 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=1)
        # X10 - 128,128

        self.deconv5 = nn.ConvTranspose2d(in_channels=128, out_channels=2, kernel_size=4, stride=2, padding=1)
        # X11 - 256, 256

    def forward(self, x):
        # Encoder
        x1 = self.pool1(nn.ReLU()(self.conv1(x)))
        x2 = self.pool2(nn.ReLU()(self.conv2(x1)))
        x3 = self.pool3_1(nn.ReLU()(self.conv3_1(self.conv3(x2))))
        x4 = self.pool4_1(nn.ReLU()(self.conv4_1(self.conv4(x3))))
        x5 = self.pool5_1(nn.ReLU()(self.conv5_1(self.conv5(x4))))
        x6 = nn.ReLU()(self.conv6(x5))

        # Decoder
        x7 = self.deconv1(x6)
        x8 = self.deconv2((torch.cat((x7, x4), dim=1)))
        x9 = self.deconv3(nn.ReLU()(torch.cat((x8, x3), dim=1)))
        x10 = self.deconv4(nn.ReLU()(torch.cat((x9, x2), dim=1)))
        x11 = self.deconv5(nn.ReLU()(torch.cat((x10, x1), dim=1)))
        return x11