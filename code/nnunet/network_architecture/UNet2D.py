import torch
import random
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
    
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet2D(nn.Module):
    def __init__(self, in_ch, out_ch, n1 = 64):
        super(UNet2D, self).__init__()


        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.Conv_ds5 = nn.Conv2d(filters[3], out_ch, kernel_size=1, stride=1, padding=0)
        self.Conv_ds4 = nn.Conv2d(filters[2], out_ch, kernel_size=1, stride=1, padding=0)
        self.Conv_ds3 = nn.Conv2d(filters[1], out_ch, kernel_size=1, stride=1, padding=0)

        self.active = nn.Softmax(dim=1)
        self.STN = STN()

    # def forward(self, x, train_img, motion_field, num_board): # x:[bs, in_ch, H, W]
    def forward(self, x):  # x:[bs, in_ch, H, W]
        out_lists = []

        e1 = self.Conv1(x) # [bs, n1, H, W]

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2) # [bs, n1*2, H/2, W/2]

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3) # [bs, n1*4, H/4, W/4]

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4) # [bs, n1*8, H/8, W/8]

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5) # [bs, n1*16, H/16, W/16]

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5) # [bs, n1*8, H/8, W/8]
        out_lists.append(self.active(self.Conv_ds5(d5)))    #deep supervision

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4) # [bs, n1*4, H/4, W/4]
        out_lists.append(self.active(self.Conv_ds4(d4)))

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3) # [bs, n1*2, H/2, H/2]
        out_lists.append(self.active(self.Conv_ds3(d3)))

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2) # [bs, n1, H, W]

        d1 = self.Conv(d2) # [bs, out_ch, H, W]
        out = self.active(d1)

        return out, out_lists


class UNet2DEncoder(nn.Module):
    def __init__(self, in_ch,n1 = 64):
        super(UNet2DEncoder, self).__init__()


        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

    def forward(self, x): # [bs, in_ch, H, W]

        e1 = self.Conv1(x) # [bs, n1, H, W]

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2) # [bs, n1*2, H/2, W/2]

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3) # [bs, n1*4, H/4, W/4]]

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4) # [bs, n1*8, H/8, W/8]

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5) # [bs, n1*16, H/16, W/16]
        
        return e1, e2, e3, e4, e5


class UNet2DDecoder(nn.Module):
    def __init__(self, out_ch,n1 = 64):
        super(UNet2DDecoder, self).__init__()


        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.active = nn.Softmax(dim=1)

    def forward(self, e):

        d5 = self.Up5(e[4])
        d5 = torch.cat((e[3], d5), dim=1)
        d5 = self.Up_conv5(d5) # [bs, n1*8, H/8, W/8]

        d4 = self.Up4(d5)
        d4 = torch.cat((e[2], d4), dim=1)
        d4 = self.Up_conv4(d4) # [bs, n1*4, H/4, W/4]

        d3 = self.Up3(d4)
        d3 = torch.cat((e[1], d3), dim=1)
        d3 = self.Up_conv3(d3) # [bs, n1*2, H/2, H/2]

        d2 = self.Up2(d3)
        d2 = torch.cat((e[0], d2), dim=1)
        d2 = self.Up_conv2(d2) # [bs, n1, H, W]

        d1 = self.Conv(d2) # [bs, out_ch, H, W]
        out = self.active(d1) 
        
        return out


class UNet2DDecoderPlus(nn.Module):
    def __init__(self, out_ch,n1 = 64):
        super(UNet2DDecoderPlus, self).__init__()


        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Up5 = up_conv(2 * filters[4], filters[3])
        self.Up_conv5 = conv_block(3 * filters[3], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(3 * filters[2], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(3 * filters[1], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(3 * filters[0], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.active = nn.Tanh()

    def forward(self, e_ref, e):

        d5 = torch.cat((e_ref[4], e[4]), dim=1)
       
        d5 = self.Up5(d5)
        d5 = torch.cat((e_ref[3], e[3], d5), dim=1)
        d5 = self.Up_conv5(d5) # [bs, n1*8, H/8, W/8]

        d4 = self.Up4(d5)
        d4 = torch.cat((e_ref[2], e[2], d4), dim=1)
        d4 = self.Up_conv4(d4) # [bs, n1*4, H/4, W/4]

        d3 = self.Up3(d4)
        d3 = torch.cat((e_ref[1], e[1], d3), dim=1)
        d3 = self.Up_conv3(d3) # [bs, n1*2, H/2, H/2]

        d2 = self.Up2(d3)
        d2 = torch.cat((e_ref[0], e[0], d2), dim=1)
        d2 = self.Up_conv2(d2) # [bs, n1, H, W]

        d1 = self.Conv(d2) # [bs, out_ch, H, W]
        out = self.active(d1) 
        
        return out


class STN(object):
    def __init__(self, mode='bilinear', isCUDA=True):
        self.mode = mode
        self.isCUDA = isCUDA
        # when input is 5D the mode='bilinear' is used as trilinear

    def __call__(self, source, offset):
        # source: NCHW
        # grid: NHW2

        x_shape = source.size()
        grid_w, grid_h = torch.meshgrid(
            [torch.linspace(-1, 1, x_shape[2]), torch.linspace(-1, 1, x_shape[3])])  # (w, h)

        if self.isCUDA == True:
            grid_w = grid_w.float().cuda()
            grid_h = grid_h.float().cuda()
        else:
            grid_w = grid_w.float()
            grid_h = grid_h.float()

        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)

        offset_w, offset_h = torch.split(offset, 1, 1)
        offset_w = offset_w.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, w, h)
        offset_h = offset_h.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, w, h)

        offset_w = grid_w + offset_w
        offset_h = grid_h + offset_h

        grid = torch.stack((offset_h, offset_w), 3)  # should have the same order as offset

        out = F.grid_sample(source, grid, align_corners=True, mode=self.mode)

        return out