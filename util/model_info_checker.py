# Model Info checker
"""
This util is developed by Suanfamama.

Authors:
- Wei Jiang (wei@suanfamama.com)
- Mama Xiao (mama.xiao@suanfamama.com)
"""
import torch
import torch.nn as nn

class MamaNet(nn.Module):
    def __init__(self):
        super(MamaNet, self).__init__()

        self.encoder1 = self.double_conv(3, 64)
        self.encoder2 = self.double_conv(64, 128)
        self.encoder3 = self.double_conv(128, 256)
        self.encoder4 = self.double_conv(256, 512)
        self.bottleneck = self.double_conv(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.double_conv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.double_conv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.double_conv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.double_conv(128, 64)

        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.downsample(enc1))
        enc3 = self.encoder3(self.downsample(enc2))
        enc4 = self.encoder4(self.downsample(enc3))
        bottleneck = self.bottleneck(self.downsample(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.final_conv(dec1))

    def downsample(self, x):
        return nn.MaxPool2d(kernel_size=2, stride=2)(x)


# 1. Load the State Dict
state_dict = torch.load('../saved_models/MamaNet_epoch_2.pth')

# 2. Instantiate the model
model = MamaNet()

# 3. Load the State Dict into the model
model.load_state_dict(state_dict)

# 4. Set to Eval Mode
model.eval()

# 5. Calculate the number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# 6. Print the result
print(f"Total number of parameters: {total_params}")

'''
Total number of parameters: 31,031,875 # 0.03B
'''