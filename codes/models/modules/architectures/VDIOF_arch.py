import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as O
from . import block as B
from . import RRDBNet_arch as R
from models.modules.architectures.video import optical_flow_warp

def vertical_upscale(x, upfield=True):

    n, c, h, w = x.shape
    h *= 2
    t = x
    if upfield:
        out = torch.cat([t, torch.zeros_like(t)], 3)
    else:
        out = torch.cat([torch.zeros_like(t), t], 3)
    out = torch.reshape(out, (n, c, -1, w))
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
        SR.append(CasResB(3, channels))
        SR.append(nn.Conv2d(channels, 64 * 1, 1, 1, 0, bias=False))
        SR.append(nn.LeakyReLU(0.1, inplace=True))
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
        optical_flow_L1_upscaled = F.interpolate(optical_flow_L1, size=(image_shape[2],image_shape[3]), mode='bilinear', align_corners=False) * 2

        #Part 2
        x_L2 = optical_flow_warp(torch.unsqueeze(x[:, 0, :, :], 1), optical_flow_L1_upscaled)
        input_L2 = torch.cat((x_L2, torch.unsqueeze(x[:, 1, :, :], 1), optical_flow_L1_upscaled), 1)
        optical_flow_L2 = self.RNN2(self.RNN1(input_L2)) + optical_flow_L1_upscaled

        #Part 3
        x_L3 = optical_flow_warp(torch.unsqueeze(x[:, 0, :, :], 1), optical_flow_L2)
        input_L3 = torch.cat((x_L3, torch.unsqueeze(x[:, 1, :, :], 1), optical_flow_L2), 1)

        optical_flow_L3 = self.SR(self.RNN1(input_L3)) + optical_flow_L2
        return optical_flow_L1, optical_flow_L2, optical_flow_L3


class VDIOFNet(nn.Module):
    '''
    Video Deinterlacing with Inpainting and Optical Flow Warping (VDIOF)
    '''
    def __init__(self, in_nc=3, out_nc=3, nf=64, act_type='leakyrelu', channels=320, n_frames=3):
        super(VDIOFNet, self).__init__()

        # Inpaint step
        # We essentially create a mini ESRGAN arch here with an upcomv block instead of an RRDB block
        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        # RRDB = R.RRDB(nf) could potentially use this later
        # LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None, mode='CNA')
        upsample = B.upconv_block(nf, nf, upscale_factor=(2, 1), kernel_size=3, act_type=act_type)
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)
        self.inpaint = B.sequential(fea_conv, upsample, HR_conv0, HR_conv1) # B.ShortcutBlock(B.sequential(RRDB, LR_conv))

        # Optical Flow Estimation Step
        # Use motion estimation to restore center frame
        self.OFR = OFRnet(320, 3)

        sr_in_nc=in_nc*(1**2 * (n_frames-1) +1)

        # Final conv step
        # Hopefully remove any residual artifacts from the deinterlacing
        clean_fea_conv = B.conv_block(sr_in_nc, nf, kernel_size=3, norm_type=None, act_type=act_type)
        clean_HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        clean_HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)
        self.clean = B.sequential(clean_fea_conv, clean_HR_conv0, clean_HR_conv1)


        
    def forward(self, x):
        # x: N, C, T, H, W
        b, n_frames, c, h, w = x.size()
        output = torch.zeros(b, n_frames, c, h*2, w).to(x.device)
        # Replace each frame with an inpainted/sr'd version of the frame
        for i in range(n_frames):
            output[:, i, :, :, :] = self.inpaint(x[:, i, :, :, :])

        idx_center = (n_frames - 1) // 2

        # motion estimation
        flow_L1 = []
        flow_L2 = []
        flow_L3 = []
        input = []

        for idx_frame in range(n_frames):
            if idx_frame != idx_center:
                input.append(torch.cat((output[:,idx_frame,:,:,:], output[:,idx_center,:,:,:]), 1))
        optical_flow_L1, optical_flow_L2, optical_flow_L3 = self.OFR(torch.cat(input, 0))

        optical_flow_L1 = optical_flow_L1.view(-1, b, 2, h//2, w//2)
        optical_flow_L2 = optical_flow_L2.view(-1, b, 2, h, w)
        optical_flow_L3 = optical_flow_L3.view(-1, b, 2, h, w)

        # motion compensation
        draft_cube = []
        draft_cube.append(output[:, idx_center, :, :, :])

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
                        draft = optical_flow_warp(output[:, idx_frame, :, :, :],
                                                  optical_flow_L3[idx, :, :, i::1, j::1] / 1)
                        draft_cube.append(draft)
        draft_cube = torch.cat(draft_cube, 1)
        # print('draft_cube:', draft_cube.shape) #TODO

        output = self.clean(draft_cube)

        return flow_L1, flow_L2, flow_L3, output
        # return output[:, idx_center, :, :, :]

        # # y_full = replace_field(y, x, upfield=True)
        # # z_full = replace_field(z, x, upfield=False)

        # return y_full, z_full


class VDSPCNet_og(nn.Module):
    '''
    Video Deinterlacing with Sub-Pixel Convolution (VDSPC)
    '''
    def __init__(self, in_nc=3, out_nc=3, nf=64, act_type='leakyrelu', n_frames=3):
        super(VDSPCNet, self).__init__()

        # Inpaint step
        # We essentially create a mini ESRGAN arch here with an upcomv block instead of an RRDB block
        fea_conv = B.conv_block(in_nc*n_frames, nf, kernel_size=3, norm_type=None, act_type=None)
        # RRDB = R.RRDB(nf) could potentially use this later
        # LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None, mode='CNA')
        upsample = B.upconv_block(nf, nf, upscale_factor=(2, 1), kernel_size=3, act_type=act_type)
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)
        self.inpaint = B.sequential(fea_conv, upsample, HR_conv0, HR_conv1) # B.ShortcutBlock(B.sequential(RRDB, LR_conv))
        
    def forward(self, x):
        # x: N, C, T, H, W
        n, c, t, h, w = x.shape
        x = x.view(n,c*t,h,w) # N, CT, H, W
        output = self.inpaint(x)

        return output

class VDSPCNet_og(nn.Module):
    '''
    Video Deinterlacing with Sub-Pixel Convolution (VDSPC)
    '''
    def __init__(self, in_nc=3, out_nc=3, nf=64, act_type='leakyrelu', n_frames=3):
        super(VDSPCNet_og, self).__init__()

        # Inpaint step
        # We essentially create a mini ESRGAN arch here with an upcomv block instead of an RRDB block
        fea_conv = B.conv_block(in_nc*n_frames, nf, kernel_size=3, norm_type=None, act_type=None)
        # RRDB = R.RRDB(nf) could potentially use this later
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None, mode='CNA')
        LR_conv_2 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None, mode='CNA')
        LR_conv_3 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None, mode='CNA')
        # upsample = B.upconv_block(nf, nf, upscale_factor=(2, 1), kernel_size=3, act_type=act_type)
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)
        self.inpaint = B.sequential(fea_conv, LR_conv, LR_conv_2, LR_conv_3, HR_conv0, HR_conv1) # B.ShortcutBlock(B.sequential(RRDB, LR_conv))
        
    def forward(self, x):
        # x: N, C, T, H, W
        n, c, t, h, w = x.shape
        x = x.view(n,c*t,h,w) # N, CT, H, W
        output = self.inpaint(x)

        return output

class VDSPCNet(nn.Module):
    '''
    Video Deinterlacing with Deformable Sub-Pixel Convolution (VDDSPC)
    '''
    def __init__(self, in_nc=3, deformable_groups=1, out_nc=3, nf=64, act_type='leakyrelu', n_frames=3):
        super(VDSPCNet, self).__init__()

        self.conv_offset = nn.Conv2d(
            in_nc*n_frames,
            deformable_groups * 2 * 3 * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    # def forward(self, x):
    #     offset = self.conv_offset(x)
    #     return deform_conv(x, offset, self.weight, self.stride, self.padding,
    #                        self.dilation, self.groups, self.deformable_groups)

        # Inpaint step
        #self.fea_conv = B.conv_block(in_nc*n_frames, nf, kernel_size=3, norm_type=None, act_type=None)
        self.dcn_conv = O.DeformConv2d(in_nc*n_frames, nf, kernel_size=3, padding=1)
        # RRDB = R.RRDB(nf) could potentially use this later
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None, mode='CNA')
        LR_conv_2 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None, mode='CNA')
        LR_conv_3 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None, mode='CNA')
        # upsample = B.upconv_block(nf, nf, upscale_factor=(2, 1), kernel_size=3, act_type=act_type)
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)
        self.inpaint = B.sequential(LR_conv, LR_conv_2, LR_conv_3, HR_conv0, HR_conv1) # B.ShortcutBlock(B.sequential(RRDB, LR_conv))
        
    def forward(self, x):
        # x: N, C, T, H, W
        n, c, t, h, w = x.shape
        x = x.view(n,c*t,h,w) # N, CT, H, W
        #output = self.inpaint(x)

        offset = self.conv_offset(x)
        deform = self.dcn_conv(x, offset=offset)
        output = self.inpaint(deform)

        return output

class VDOFNet_old(nn.Module):
    '''
    Video Deinterlacing with Optical Flow Warping (VDIOF)
    '''
    def __init__(self, in_nc=3, out_nc=3, nf=64, act_type='leakyrelu', channels=320, n_frames=3):
        super(VDOFNet, self).__init__()

        # Inpaint step
        # We essentially create a mini ESRGAN arch here with an upcomv block instead of an RRDB block
        # fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        # # RRDB = R.RRDB(nf) could potentially use this later
        # # LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=None, mode='CNA')
        # upsample = B.upconv_block(nf, nf, upscale_factor=(2, 1), kernel_size=3, act_type=act_type)
        # HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        # HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)
        # self.inpaint = B.sequential(fea_conv, upsample, HR_conv0, HR_conv1) # B.ShortcutBlock(B.sequential(RRDB, LR_conv))

        # Optical Flow Estimation Step
        # Use motion estimation to restore center frame
        self.OFR = OFRnet(channels, 3)

        sr_in_nc=in_nc*((n_frames-1) +1)

        body = []
        body.append(nn.Conv2d(sr_in_nc, channels, 3, 1, 1, bias=False))
        body.append(nn.LeakyReLU(0.1, inplace=True))
        body.append(CasResB(8, channels))
        body.append(nn.Conv2d(channels, 64 * 1, 1, 1, 0, bias=False))
        body.append(nn.LeakyReLU(0.1, inplace=True))
        body.append(nn.Conv2d(64, out_nc, 3, 1, 1, bias=True))
        self.draft_cube_conv = nn.Sequential(*body)

        # sr_in_nc=in_nc*(1**2 * (n_frames-1) +1)

        # Final conv step
        # Hopefully remove any residual artifacts from the deinterlacing
        clean_fea_conv = B.conv_block(in_nc, in_nc*2, kernel_size=3, norm_type=None, act_type=act_type)
        clean_HR_conv0 = B.conv_block(in_nc*2, in_nc*2, kernel_size=3, norm_type=None, act_type=act_type)
        clean_HR_conv1 = B.conv_block(in_nc*2, out_nc, kernel_size=3, norm_type=None, act_type=None)
        self.clean = B.sequential(clean_fea_conv, clean_HR_conv0, clean_HR_conv1)

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
        optical_flow_L3 = optical_flow_L3.view(-1, b, 2, h, w)

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

        #output = self.clean(draft_cube)

        return flow_L1, flow_L2, flow_L3, draft_cube

        
    def forward(self, x):
        # x: N, C, T, H, W
        b, n_frames, c, h, w = x.size()

        idx_center = (n_frames - 1) // 2
        even_center_field = torch.zeros(b, n_frames, c, h//2, w).to(x.device)
        odd_center_field = torch.zeros(b, n_frames, c, h//2, w).to(x.device)
        for i_frame in range(n_frames):
            odd_center_field[:, i_frame, :, i_frame%2::2, :]
            even_center_field[:, i_frame, :, (i_frame+1)%2::2, :]

        ocf_flow_L1, ocf_flow_L2, ocf_flow_L3, ocf_draft_cube = self.motion_estimation(odd_center_field)
        ecf_flow_L1, ecf_flow_L2, ecf_flow_L3, ecf_draft_cube = self.motion_estimation(even_center_field)

        ocf = self.draft_cube_conv(ocf_draft_cube)
        ecf = self.draft_cube_conv(ecf_draft_cube)

        ocf_full = replace_field(ocf, x[:, idx_center, :, :, :], upfield=True)
        ecf_full = replace_field(ecf, x[:, idx_center, :, :, :], upfield=False)

        ocf_output = self.clean(ocf_full)
        ecf_output = self.clean(ecf_full)

        # return ocf_flow_L1, ocf_flow_L2, ocf_flow_L3, ocf_output
        
        return ocf_flow_L1, ocf_flow_L2, ocf_flow_L3, ocf_output, ecf_flow_L1, ecf_flow_L2, ecf_flow_L3, ecf_output


class VDOFNet(nn.Module):
    '''
    Video Deinterlacing with Optical Flow Warping (VDOF)
    '''
    def __init__(self, in_nc=3, out_nc=3, nf=64, act_type='leakyrelu', channels=320, n_frames=3):
        super(VDOFNet, self).__init__()

        conv_fea_1 = nn.Sequential(nn.Conv2d(in_nc*n_frames, nf, kernel_size=3, padding=1), nn.ReLU())
        conv_fea_2 = nn.Sequential(nn.Conv2d(nf, nf, kernel_size=3, padding=1), nn.ReLU())
        conv_fea_3 = nn.Conv2d(nf, nf//2, kernel_size=1)
        h = nn.Sequential(conv_fea_1, conv_fea_2, conv_fea_3)

        conv_branch_top = nn.Conv2d(nf//2, nf//2, kernel_size=3, padding=1)
        conv_branch_bottom = nn.Conv2d(nf//2, nf//2, kernel_size=3, padding=1)

        final_branch_top = nn.Conv2d(nf//2, out_nc*n_frames, kernel_size=3, stride=(2, 1), padding=1)
        final_branch_bottom = nn.Conv2d(nf//2, out_nc*n_frames, kernel_size=3, stride=(2, 1), padding=1)

        self.model_y = nn.Sequential(h, conv_branch_top, final_branch_top)
        self.model_z = nn.Sequential(h, conv_branch_bottom, final_branch_bottom)

        # Optical Flow Estimation Step
        # Use motion estimation to restore center frame
        self.OFR = OFRnet(channels, 3)

        sr_in_nc=in_nc*((n_frames-1) +1)

        body = []
        body.append(nn.Conv2d(sr_in_nc, channels, 3, 1, 1, bias=False))
        body.append(nn.LeakyReLU(0.1, inplace=True))
        body.append(CasResB(8, channels))
        body.append(nn.Conv2d(channels, 64 * 1, 1, 1, 0, bias=False))
        body.append(nn.LeakyReLU(0.1, inplace=True))
        body.append(nn.Conv2d(64, out_nc, 3, 1, 1, bias=True))
        self.draft_cube_conv = nn.Sequential(*body)

        # sr_in_nc=in_nc*(1**2 * (n_frames-1) +1)

        # Final conv step
        # Hopefully remove any residual artifacts from the deinterlacing
        # clean_fea_conv = B.conv_block(in_nc, in_nc*2, kernel_size=3, norm_type=None, act_type=act_type)
        # clean_HR_conv0 = B.conv_block(in_nc*2, in_nc*2, kernel_size=3, norm_type=None, act_type=act_type)
        # clean_HR_conv1 = B.conv_block(in_nc*2, out_nc, kernel_size=3, norm_type=None, act_type=None)
        # self.clean = B.sequential(clean_fea_conv, clean_HR_conv0, clean_HR_conv1)

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
        optical_flow_L3 = optical_flow_L3.view(-1, b, 2, h, w)

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

        #output = self.clean(draft_cube)

        return flow_L1, flow_L2, flow_L3, draft_cube

        
    def forward(self, x):
        # x: N, C, T, H, W
        b, n_frames, c, h, w = x.size()

        x = x.view(b,c*n_frames,h,w) # N, CT, H, W

        # Combines the interlaced data of each frame into a full frame
        y = self.model_y(x)
        z = self.model_z(x)
        x[:, :, 0::2, :] = y
        x[:, :, 1::2, :] = z

        x = x.view(b,c,n_frames,h,w) # N, C, T, H, W

        # Optical Flow Motion Estimation
        flow_L1, flow_L2, flow_L3, draft_cube = self.motion_estimation(x)

        # Convert draft cube into single image
        out = self.draft_cube_conv(draft_cube)

        return flow_L1, flow_L2, flow_L3, out
        