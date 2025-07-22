import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch, scale):
        super(up_conv, self).__init__()
        
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNet3D, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool3d(kernel_size=(3,2,2), padding=(1,0,0), stride=(1,2,2))
        self.Maxpool2 = nn.MaxPool3d(kernel_size=(3,2,2), padding=(1,0,0), stride=(1,2,2))
        self.Maxpool3 = nn.MaxPool3d(kernel_size=(3,2,2), padding=(1,0,0), stride=(1,2,2))
        self.Maxpool4 = nn.MaxPool3d(kernel_size=(3,2,2), padding=(1,0,0), stride=(1,2,2))

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3], (1,2,2))
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2], (1,2,2))
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1], (1,2,2))
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0], (1,2,2))
        self.Up_conv2 = conv_block(filters[1], filters[0])
        
        self.Conv = nn.Conv3d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.active = nn.Softmax(dim=1)

    def forward(self, x): # [bs, 1, 25, H, W]

        e1 = self.Conv1(x) # [bs, 64, 25, H, W]

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2) # [bs, 128, 25, H/2, W/2]

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3) # [bs, 256, 25, H/4, W/4]

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4) # [bs, 512, 25, H/8, W/8]

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5) # [bs, 1024, 25, H/16, W/16]

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5) # [bs, 512, 25, H/8, W/8]

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4) # [bs, 256, 25, H/4, W/4]

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3) # [bs, 128, 25, H/2, W/2]

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2) # [bs, 64, 25, H, W]

        d1 = self.Conv(d2) # [bs, 3, 25, H, W]
        out = self.active(d1)

        return out


class UNet3DEncoder(nn.Module):
    def __init__(self, in_ch):
        super(UNet3DEncoder, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool3d(kernel_size=(3,2,2), padding=(1,0,0), stride=(1,2,2))
        self.Maxpool2 = nn.MaxPool3d(kernel_size=(3,2,2), padding=(1,0,0), stride=(1,2,2))
        self.Maxpool3 = nn.MaxPool3d(kernel_size=(3,2,2), padding=(1,0,0), stride=(1,2,2))
        self.Maxpool4 = nn.MaxPool3d(kernel_size=(3,2,2), padding=(1,0,0), stride=(1,2,2))
        
        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

    def forward(self, x): # [bs, 1, 25, H, W]   #predict x:[bs,17,1,122,122]

        e1 = self.Conv1(x) # [bs, 64, 25, H, W]

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2) # [bs, 128, 25, H/2, W/2]

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3) # [bs, 256, 25, H/4, W/4]

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4) # [bs, 512, 25, H/8, W/8]

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5) # [bs, 1024, 25, H/16, W/16]
        
        return e1, e2, e3, e4, e5


class UNet3DDecoder(nn.Module):
    def __init__(self, out_ch):
        super(UNet3DDecoder, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Up5 = up_conv(filters[4], filters[3], (1,2,2))
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2], (1,2,2))
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1], (1,2,2))
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0], (1,2,2))
        self.Up_conv2 = conv_block(filters[1], filters[0])
        
        self.Conv = nn.Conv3d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.active = nn.Softmax(dim=1)

    def forward(self, e):

        d5 = self.Up5(e[4])
        d5 = torch.cat((e[3], d5), dim=1)
        d5 = self.Up_conv5(d5) # [bs, 512, 25, H/8, W/8]

        d4 = self.Up4(d5)
        d4 = torch.cat((e[2], d4), dim=1)
        d4 = self.Up_conv4(d4) # [bs, 256, 25, H/4, W/4]

        d3 = self.Up3(d4)
        d3 = torch.cat((e[1], d3), dim=1)
        d3 = self.Up_conv3(d3) # [bs, 128, 25, H/2, W/2]
        
        d2 = self.Up2(d3)
        d2 = torch.cat((e[0], d2), dim=1)
        d2 = self.Up_conv2(d2) # [bs, 64, 25, H, W]

        d1 = self.Conv(d2) # [bs, 3, 25, H, W]
        out = self.active(d1) 
        
        return out


class UNet3DDecoderPlus(nn.Module):
    def __init__(self, out_ch):
        super(UNet3DDecoderPlus, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Up5 = up_conv(2 * filters[4], filters[3], (1,2,2))
        self.Up_conv5 = conv_block(3 * filters[3], filters[3])

        self.Up4 = up_conv(filters[3], filters[2], (1,2,2))
        self.Up_conv4 = conv_block(3 * filters[2], filters[2])

        self.Up3 = up_conv(filters[2], filters[1], (1,2,2))
        self.Up_conv3 = conv_block(3 * filters[1], filters[1])

        self.Up2 = up_conv(filters[1], filters[0], (1,2,2))
        self.Up_conv2 = conv_block(3 * filters[0], filters[0])
        
        self.Conv = nn.Conv3d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
    
    def forward(self, e_ref, e):

        d5 = torch.cat((e_ref[4], e[4]), dim=1)
       
        d5 = self.Up5(d5)
        d5 = torch.cat((e_ref[3], e[3], d5), dim=1)
        d5 = self.Up_conv5(d5) # [bs, 512, 25, H/8, W/8]

        d4 = self.Up4(d5)
        d4 = torch.cat((e_ref[2], e[2], d4), dim=1)
        d4 = self.Up_conv4(d4) # [bs, 256, 25, H/4, W/4]

        d3 = self.Up3(d4)
        d3 = torch.cat((e_ref[1], e[1], d3), dim=1)
        d3 = self.Up_conv3(d3) # [bs, 128, 25, H/2, W/2]

        d2 = self.Up2(d3)
        d2 = torch.cat((e_ref[0], e[0], d2), dim=1)
        d2 = self.Up_conv2(d2) # [bs, 64, 25, H, W]

        d1 = self.Conv(d2) # [bs, 3, 25, H, W]

        return d1
