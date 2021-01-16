import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as O
from . import block as B
from . import RRDBNet_arch as R
from models.modules.architectures.video import optical_flow_warp

def channel_shuffle(x, groups):
    # print(x.size()) #TODO
    b, c, h, w = x.size()
    x = x.view(b, groups, c//groups,  h, w)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    x = x.view(b, -1, h, w)
    return x

class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels//2, channels//2, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels//2, channels//2, 3, 1, 1, bias=False, groups=channels//2),
            nn.Conv2d(channels // 2, channels // 2, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
    def forward(self, x):
        input = x[:, x.shape[1]//2:, :, :]
        out = torch.cat((x[:, :x.shape[1]//2, :, :], self.body(input)), 1)
        return channel_shuffle(out, 2)


class CasResB(nn.Module):
    def __init__(self, n_ResB, channels):
        super(CasResB, self).__init__()
        body = []
        for i in range(n_ResB):
            body.append(ResB(channels))
        self.body = nn.Sequential(*body)
    def forward(self, x):
        return self.body(x)

class OFRnet(nn.Module):
    def __init__(self, channels, img_ch=3):
        super(OFRnet, self).__init__()
        self.pool = nn.AvgPool2d(2)

        ## RNN part
        self.RNN1 = nn.Sequential(
            nn.Conv2d(2*(img_ch+1), channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            CasResB(3, channels)
        )
        self.RNN2 = nn.Sequential(
            nn.Conv2d(channels, 2*img_ch, 3, 1, 1, bias=False),
        )

        SR = []
        # SR.append(nn.Conv2d(channels, 64 * 4, 1, 1, 0, bias=False))
        # SR.append(nn.PixelShuffle(2)) #TODO
        # SR.append(nn.LeakyReLU(0.1, inplace=True))
        # Switch from pixelshuffle to upconv block here
        SR.append(B.upconv_block(channels, 64, upscale_factor=1, kernel_size=3, stride=1, act_type='leakyrelu'))
        SR.append(nn.Conv2d(64, 2*img_ch, 3, 1, 1, bias=False))
        self.SR = nn.Sequential(*SR)

    def __call__(self, x):
        # x: b*2*h*w
        #Part 1
        x_L1 = self.pool(x)
        b, c, h, w = x_L1.size()
        input_L1 = torch.cat((x_L1, torch.zeros(b, 2, h, w).cuda()), 1)
        one = self.RNN1(input_L1)
        optical_flow_L1 = self.RNN2(one)

        image_shape = torch.unsqueeze(x[:, 0, :, :], 1).shape
        optical_flow_L1_upscaled = F.interpolate(optical_flow_L1, size=(image_shape[2],image_shape[3]), mode='bilinear', align_corners=False) * 1

        #Part 2
        x_L2 = optical_flow_warp(torch.unsqueeze(x[:, 0, :, :], 1), optical_flow_L1_upscaled)
        input_L2 = torch.cat((x_L2, torch.unsqueeze(x[:, 1, :, :], 1), optical_flow_L1_upscaled), 1)
        optical_flow_L2 = self.RNN2(self.RNN1(input_L2)) + optical_flow_L1_upscaled

        #Part 3
        x_L3 = optical_flow_warp(torch.unsqueeze(x[:, 0, :, :], 1), optical_flow_L2)
        input_L3 = torch.cat((x_L3, torch.unsqueeze(x[:, 1, :, :], 1), optical_flow_L2), 1)

        optical_flow_L3 = self.SR(self.RNN1(input_L3)) + F.interpolate(optical_flow_L2, scale_factor=1, mode='bilinear', align_corners=False) * 1
        return optical_flow_L1, optical_flow_L2, optical_flow_L3

        

class VDOFNet(nn.Module):
    '''
    Video Deinterlacing with Optical Flow Warping (VDOF)
    SOF-VDI
    '''
    def __init__(self, in_nc=3, out_nc=3, nf=64, act_type='leakyrelu', channels=320, n_frames=3):
        super(VDOFNet, self).__init__()

        # Optical Flow Estimation Step
        # Use motion estimation to restore center frame
        self.OFR = OFRnet(channels, 3)

        sr_in_nc=in_nc*((1**2) * (n_frames-1) +1)

        body = []
        body.append(nn.Conv2d(sr_in_nc, channels, 3, 1, 1, bias=False))
        body.append(nn.LeakyReLU(0.1, inplace=True))
        body.append(CasResB(8, channels))
        # body.append(nn.Conv2d(channels, 64 * 4, 1, 1, 0, bias=False))
        # body.append(nn.PixelShuffle(2)) #TODO
        # body.append(nn.LeakyReLU(0.1, inplace=True))
        body.append(B.upconv_block(channels, 64, upscale_factor=1, kernel_size=3, stride=1, act_type='leakyrelu'))
        body.append(nn.Conv2d(64, out_nc, 3, 1, 1, bias=True))
        self.draft_cube_conv = nn.Sequential(*body)

    def motion_estimation(self, x):
        b, n_frames, c, h, w = x.size()
        idx_center = (n_frames - 1) // 2

        flow_L1 = []
        flow_L2 = []
        flow_L3 = []
        input = []

        for idx_frame in range(n_frames):
            if idx_frame != idx_center:
                input.append(torch.cat((x[:,idx_frame,:,:,:], x[:,idx_center,:,:,:]), 1))
        optical_flow_L1, optical_flow_L2, optical_flow_L3 = self.OFR(torch.cat(input, 0))

        optical_flow_L1 = optical_flow_L1.view(-1, b, 2, h//2, w//2)
        optical_flow_L2 = optical_flow_L2.view(-1, b, 2, h, w)
        optical_flow_L3 = optical_flow_L3.view(-1, b, 2, h*1, w*1)

        # motion compensation
        draft_cube = []
        draft_cube.append(x[:, idx_center, :, :, :])

        for idx_frame in range(n_frames):
            if idx_frame == idx_center:
                flow_L1.append([])
                flow_L2.append([])
                flow_L3.append([])
            else: # if idx_frame != idx_center:
                if idx_frame < idx_center:
                    idx = idx_frame
                if idx_frame > idx_center:
                    idx = idx_frame - 1

                flow_L1.append(optical_flow_L1[idx, :, :, :, :])
                flow_L2.append(optical_flow_L2[idx, :, :, :, :])
                flow_L3.append(optical_flow_L3[idx, :, :, :, :])

                # Generate the draft_cube by subsampling the SR flow optical_flow_L3
                # according to the scale
                for i in range(1):
                    for j in range(1):
                        draft = optical_flow_warp(x[:, idx_frame, :, :, :],
                                                  optical_flow_L3[idx, :, :, i::1, j::1] / 1)
                        draft_cube.append(draft)
        draft_cube = torch.cat(draft_cube, 1)
        # print('draft_cube:', draft_cube.shape) #TODO

        return flow_L1, flow_L2, flow_L3, draft_cube

        
    def forward(self, x):
        # B, T, C, H, W
        b, n_frames, c, h, w = x.size()
        center_idx = (n_frames - 1) // 2

        # Split tensor into fields
        # odd = x[:, :, :, 0::2, :]
        # even = x[:, :, :, 1::2, :]
        # Concat fields on height dimension
        # x = torch.cat((odd, even), 3) # B, T, C, H, W
        # Concat 3 frames on channel dimension
        # x = torch.cat([x[:, t, :, :, :] for t in range(n_frames)], 1) # B, C*T, H, W
        # Resize to double height 
        # x = F.interpolate(x, size=(h*2, w), mode='nearest') # B, C*T, H*2, W
        # View back to 3 concatenated fields
        # x = x.view(b, n_frames, c, h*2, w) # B, T, C, H*2, W

        # Optical Flow Motion Estimation
        flow_L1, flow_L2, flow_L3, draft_cube = self.motion_estimation(x)

        # Convert draft cube into 2 stacked images
        draft = self.draft_cube_conv(draft_cube) # [:, :, :, 0::2]
        # odd_draft = draft[:, :, :h, 0::2]
        # even_draft = draft[:, :, h:, 0::2]
        # print(odd_draft.shape)

        # out_odd = torch.zeros(b, c, h*2, w)
        # out_even = torch.zeros(b, c, h*2, w)
        # print(odd[:, center_idx, :, :, :].shape)

        # out_odd[:, :, 0::2, :] = odd[:, center_idx, :, :, :]
        # out_odd[:, :, 1::2, :] = odd_draft
        # out_even[:, :, 0::2, :] = even_draft
        # out_even[:, :, 1::2, :] = even[:, center_idx, :, :, :]

        # out = torch.cat((out_odd, out_even), 2).to(x.device)

        return flow_L1, flow_L2, flow_L3, draft
        