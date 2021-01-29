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
    def __init__(self, channels, img_ch=3, scale=1, nf=64):
        super(OFRnet, self).__init__()
        self.pool = nn.AvgPool2d(2)
        self.scale = scale

        ## RNN part
        self.RNN1 = nn.Sequential(
            nn.Conv2d(2*(img_ch+1), channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            CasResB(3, channels)
        )
        self.RNN2 = nn.Sequential(
            nn.Conv2d(channels, 2*img_ch, 3, 1, 1, bias=False),
        )

        SR = [
            # R.ResidualDenseBlock_5C(channels, kernel_size=3),
            B.upconv_block(channels, nf, upscale_factor=scale, kernel_size=3, stride=1, act_type='leakyrelu'),
            # R.ResidualDenseBlock_5C(nf, kernel_size=3),
            nn.Conv2d(nf, 2*img_ch, 3, 1, 1, bias=False),
        ]
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
        optical_flow_L1_upscaled = F.interpolate(optical_flow_L1, size=(image_shape[2],image_shape[3]), mode='bilinear', align_corners=False) * 2

        #Part 2
        x_L2 = optical_flow_warp(torch.unsqueeze(x[:, 0, :, :], 1), optical_flow_L1_upscaled)
        input_L2 = torch.cat((x_L2, torch.unsqueeze(x[:, 1, :, :], 1), optical_flow_L1_upscaled), 1)
        optical_flow_L2 = self.RNN2(self.RNN1(input_L2)) + optical_flow_L1_upscaled

        #Part 3
        x_L3 = optical_flow_warp(torch.unsqueeze(x[:, 0, :, :], 1), optical_flow_L2)
        input_L3 = torch.cat((x_L3, torch.unsqueeze(x[:, 1, :, :], 1), optical_flow_L2), 1)

        optical_flow_L3 = self.SR(self.RNN1(input_L3)) + F.interpolate(optical_flow_L2, scale_factor=self.scale, mode='bilinear', align_corners=False) * self.scale
        return optical_flow_L1, optical_flow_L2, optical_flow_L3

        

class VDOFNet(nn.Module):
    '''
    Video Deinterlacing with Optical Flow Warping (VDOF)
    RS-OF-VDI (Residual Optical Flow Video De-Interlacing)
    '''
    def __init__(self, in_nc=3, out_nc=3, nf=64, act_type='leakyrelu', channels=320, n_frames=3):
        super(VDOFNet, self).__init__()

        self.scale = 2

        # Bring resized and aligned fields back to half height
        squash = [
            nn.Conv2d(in_nc*n_frames, nf, 3, 1, 1, bias=True),
            # R.ResidualDenseBlock_5C(nf, kernel_size=3),
            nn.Conv2d(nf, nf, 3, (2, 1), 1, bias=True),
            nn.Conv2d(nf, in_nc*n_frames, 3, 1, 1, bias=True),
        ]
        self.squash = nn.Sequential(*squash)

        # Optical Flow Estimation Step
        # Use motion estimation to restore center frame
        self.OFR = OFRnet(channels, 3, scale=self.scale, nf=nf)

        # Calculate number of channels for SR input
        sr_in_nc=in_nc*((self.scale**2) * (n_frames-1) +1)

        SR = [
            nn.Conv2d(sr_in_nc, channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            CasResB(8, channels), 
            # R.ResidualDenseBlock_5C(channels, kernel_size=3),
            B.upconv_block(channels, nf, upscale_factor=self.scale, kernel_size=3, stride=1, act_type=act_type),
            # R.ResidualDenseBlock_5C(nf, kernel_size=3),
            B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type),
            nn.Conv2d(nf, out_nc, 3, (1, 2), 1, bias=True), # half-width output
        ]
        self.draft_cube_conv = nn.Sequential(*SR)

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
        optical_flow_L3 = optical_flow_L3.view(-1, b, 2, h*self.scale, w*self.scale)

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
                for i in range(self.scale):
                    for j in range(self.scale):
                        draft = optical_flow_warp(x[:, idx_frame, :, :, :],
                                                  optical_flow_L3[idx, :, :, i::self.scale, j::self.scale] / self.scale)
                        draft_cube.append(draft)
        draft_cube = torch.cat(draft_cube, 1)
        # print('draft_cube:', draft_cube.shape) #TODO

        return flow_L1, flow_L2, flow_L3, draft_cube

        
    def forward(self, x):
        # B, T, C, H, W
        b, n_frames, c, h, w = x.size()
        center_idx = (n_frames - 1) // 2

        # Cat frames into channel dimension
        x = x.view(b, n_frames*c, h, w) # B, CT, H, W 
        # Squash into half-height fields using strided convolution
        x = self.squash(x)
        # Expand back into frame dimension
        x = x.view(b, n_frames, c, h//2, w) # B, C, T, H//2, W 

        # Optical Flow Motion Estimation
        flow_L1, flow_L2, flow_L3, draft_cube = self.motion_estimation(x)

        # Convert draft cube into image
        draft = self.draft_cube_conv(draft_cube)

        return flow_L1, flow_L2, flow_L3, draft
        