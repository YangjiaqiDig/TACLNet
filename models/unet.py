import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
        nn.ReLU(inplace=True),
    )


class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()

        self.conv1_block = double_conv(1, 32)
        self.conv2_block = double_conv(32, 64)
        self.conv3_block = double_conv(64, 128)
        self.conv4_block = double_conv(128, 256)
        self.conv5_block = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.conv_up_1 = double_conv(512, 256)

        self.up_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv_up_2 = double_conv(256, 128)

        self.up_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2)  # for 1250 * 1250 kernel_size=3, stride=2  ## for 1024 * 1024 kernel_size=2, stride=2
        self.conv_up_3 = double_conv(128, 64)

        self.up_4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.conv_up_4 = double_conv(64, 32)

        self.conv_final = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1, padding=0, stride=1)
        self.softmax = nn.Softmax2d()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Down 1
        conv1 = self.conv1_block(x)
        x = self.maxpool(conv1)
        # Down 2
        conv2 = self.conv2_block(x)
        x = self.maxpool(conv2)

        # Down 3
        conv3 = self.conv3_block(x)
        x = self.maxpool(conv3)

        # # # Down 4
        conv4 = self.conv4_block(x)
        x = self.maxpool(conv4)

        # # Midpoint
        x = self.conv5_block(x)


        # Up 1
        x = self.up_1(x)
        x = torch.cat([x, conv4], dim=1)

        x = self.conv_up_1(x)

        # # Up 2
        x = self.up_2(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv_up_2(x)

        # Up 3
        x = self.up_3(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv_up_3(x)

        # Up 4
        x = self.up_4(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv_up_4(x)

        # Final output
        x = self.conv_final(x)
        # print('final: ', x)
        likelihood_map = self.softmax(x)[:,1,:,:]
        # print(likelihood_map)

        return x, likelihood_map


if __name__ == "__main__":
    # A full forward pass
    im = torch.randn(2, 1, 256, 256)
    model = UNET()
    likelihood_map = model(im)
    # print(x, likelihood_map)
    # print(x.shape)
    del model
    del likelihood_map
    # print(x.shape)
