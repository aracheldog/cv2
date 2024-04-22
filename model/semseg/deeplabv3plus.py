from model.semseg.base import BaseNet

import torch
from torch import nn
import torch.nn.functional as F

import torch
from torch import nn
import torch.nn.functional as F
from model.semseg.base import BaseNet

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        return out

class DeepLabV3Plus(BaseNet):
    def __init__(self, backbone, nclass):
        super(DeepLabV3Plus, self).__init__(backbone)

        # low_level_channels = 4 * 64 = 256
        low_level_channels = self.backbone.channels[0]
        #  4 * 512 = 2058
        high_level_channels = self.backbone.channels[-1]

        self.head = ASPPModule(high_level_channels, (12, 24, 36))

        self.reduce = nn.Sequential(nn.Conv2d(low_level_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.attention = SelfAttention(high_level_channels // 8)

        # self.fuse = nn.Sequential(nn.Conv2d(high_level_channels // 8 + 48, 256, 3, padding=1, bias=False),
        #                           nn.BatchNorm2d(256),
        #                           nn.ReLU(True),
        #
        #                           nn.Conv2d(256, 256, 3, padding=1, bias=False),
        #                           nn.BatchNorm2d(256),
        #                           nn.ReLU(True),
        #                           nn.Dropout(0.1, False))
        self.fuse = nn.Sequential(
            nn.Conv2d((high_level_channels // 8 + 48), 256, 3, padding=1, bias=False),  # Adjusted input channels
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout(0.1, False)
        )

        self.classifier = nn.Conv2d(256, nclass, 1, bias=True)

    def base_forward(self, x):
        h, w = x.shape[-2:]

        c1, _, _, c4 = self.backbone.base_forward(x)

        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)

        c4_attended = self.attention(c4)
        out = torch.cat([c1, c4], dim=1)
        out = self.fuse(out)
        out = self.classifier(out)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

        return out


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True),
                                     nn.Dropout2d(0.5, False))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)



#
# from model.semseg.base import BaseNet
#
# import torch
# from torch import nn
# import torch.nn.functional as F
#
#
# class DeepLabV3Plus(BaseNet):
#     def __init__(self, backbone, nclass):
#         super(DeepLabV3Plus, self).__init__(backbone)
#
#         low_level_channels = self.backbone.channels[0]
#         high_level_channels = self.backbone.channels[-1]
#
#         self.head = ASPPModule(high_level_channels, (12, 24, 36))
#
#         self.reduce = nn.Sequential(nn.Conv2d(low_level_channels, 48, 1, bias=False),
#                                     nn.BatchNorm2d(48),
#                                     nn.ReLU(True))
#
#         self.fuse = nn.Sequential(nn.Conv2d(high_level_channels // 8 + 48, 256, 3, padding=1, bias=False),
#                                   nn.BatchNorm2d(256),
#                                   nn.ReLU(True),
#
#                                   nn.Conv2d(256, 256, 3, padding=1, bias=False),
#                                   nn.BatchNorm2d(256),
#                                   nn.ReLU(True),
#                                   nn.Dropout(0.1, False))
#
#         self.classifier = nn.Conv2d(256, nclass, 1, bias=True)
#
#     def base_forward(self, x):
#         h, w = x.shape[-2:]
#
#         c1, _, _, c4 = self.backbone.base_forward(x)
#
#         c4 = self.head(c4)
#         c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)
#
#         c1 = self.reduce(c1)
#
#         out = torch.cat([c1, c4], dim=1)
#         out = self.fuse(out)
#
#         out = self.classifier(out)
#         out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
#
#         return out
#
#
# def ASPPConv(in_channels, out_channels, atrous_rate):
#     block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
#                                     dilation=atrous_rate, bias=False),
#                           nn.BatchNorm2d(out_channels),
#                           nn.ReLU(True))
#     return block
#
#
# class ASPPPooling(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ASPPPooling, self).__init__()
#         self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
#                                  nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                  nn.BatchNorm2d(out_channels),
#                                  nn.ReLU(True))
#
#     def forward(self, x):
#         h, w = x.shape[-2:]
#         pool = self.gap(x)
#         return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)
#
#
# class ASPPModule(nn.Module):
#     def __init__(self, in_channels, atrous_rates):
#         super(ASPPModule, self).__init__()
#         out_channels = in_channels // 8
#         rate1, rate2, rate3 = atrous_rates
#
#         self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
#                                 nn.BatchNorm2d(out_channels),
#                                 nn.ReLU(True))
#         self.b1 = ASPPConv(in_channels, out_channels, rate1)
#         self.b2 = ASPPConv(in_channels, out_channels, rate2)
#         self.b3 = ASPPConv(in_channels, out_channels, rate3)
#         self.b4 = ASPPPooling(in_channels, out_channels)
#
#         self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
#                                      nn.BatchNorm2d(out_channels),
#                                      nn.ReLU(True),
#                                      nn.Dropout2d(0.5, False))
#
#     def forward(self, x):
#         feat0 = self.b0(x)
#         feat1 = self.b1(x)
#         feat2 = self.b2(x)
#         feat3 = self.b3(x)
#         feat4 = self.b4(x)
#         y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
#         return self.project(y)