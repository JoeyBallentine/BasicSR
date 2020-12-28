from collections import OrderedDict

import torch
import torch.nn as nn


def vertical_upscale(x, upfield=True):
    n, c, h, w = x.shape
    h *= 2
    t = x
    if upfield:
        out = torch.cat([t, torch.zeros_like(t)], 3)
    else:
        out = torch.cat([torch.zeros_like(t), t], 3)
    out = torch.reshape(out, (n, c, h, w))
    return out


def replace_field(x, input_image, upfield=True):
    upper_input = input_image[:, :, 0::2, :]
    lower_input = input_image[:, :, 1::2, :]
    # print(upper_input.shape, lower_input.shape)

    if upfield:
        # print(upper_input.shape, x.shape)
        x = vertical_upscale(x, upfield=False)
        upper_input = vertical_upscale(upper_input, upfield=True)
        # print(upper_input.shape, x.shape)
        out = x + upper_input
    else:
        x = vertical_upscale(x, upfield=True)
        lower_input = vertical_upscale(lower_input, upfield=False)
        out = x + lower_input

    return out


class DVDNet(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64):
        """
        Real-time Deep Video Deinterlacing: https://arxiv.org/pdf/1708.00187.pdf
        Original (community-made) tensorflow code: https://github.com/lszhuhaichao/Deep-Video-Deinterlacing

        :param in_nc: Input number of channels
        :param out_nc: Output number of channels
        :param nf: Number of filters
        """
        super(DVDNet, self).__init__()
        conv_branch = final_branch = OrderedDict(top=None, bottom=None)

        conv_fea_1 = nn.Sequential(nn.Conv2d(in_nc, nf, 3), nn.ReLU())
        conv_fea_2 = nn.Sequential(nn.Conv2d(nf, nf, 3), nn.ReLU())
        conv_fea_3 = nn.Conv2d(nf, nf // 2, 1)
        h = nn.Sequential(conv_fea_1, conv_fea_2, conv_fea_3)

        for field_order in conv_branch.keys():
            conv_branch[field_order] = nn.Conv2d(nf // 2, nf // 2, 3)

        for field_order in final_branch.keys():
            final_branch[field_order] = nn.Conv2d(
                in_channels=nf // 2,
                out_channels=out_nc,
                kernel_size=3,
                stride=(2, 1),
                padding=4,
                padding_mode='replicate'
            )

        self.model_y = nn.Sequential(h, conv_branch["top"], final_branch["top"])
        self.model_z = nn.Sequential(h, conv_branch["bottom"], final_branch["bottom"])

    def forward(self, x):
        return (
            replace_field(self.model_y(x), x, upfield=True),
            replace_field(self.model_z(x), x, upfield=False)
        )
