import torch
import torch.functional as F
import torch.nn as nn

class FlowNetS(torch.nn.Module):

    def __init__(self):
        super(FlowNet, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=7, stride=2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=2)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)

        self.conv3_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3_1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=2)

        self.conv4_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.pool4_1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.pool5_1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=2)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        # Encoder
        x1 = self.pool1((self.conv1(x)))
        x2 = self.pool2((self.conv2(x1)))
        x3 = self.pool3_1((self.conv3_1(self.conv3(x2))))
        x4 = self.pool4_1((self.conv4_1(self.conv4(x3))))
        x5 = self.pool5_1((self.conv5_1(self.conv5(x4))))
        x6 = nn.ReLU()(self.conv6(x5))

        # Decoder
        x7 = self.deconv1(x6)
        x8 = self.deconv2(nn.ReLU()(torch.cat((x7, x5), 1)))  # Concatenate with encoder output
        x9 = self.deconv3(nn.ReLU()(torch.cat((x8, x4), 1)))  # Concatenate with encoder output
        # Continue with additional decoder layers...

        return x9  # The final output of the decoder

# Create an instance of the FlowNet model
model = FlowNet()


    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x