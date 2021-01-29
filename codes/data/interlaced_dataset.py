import os.path as path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import dataops.common as util
import dataops.augmentations as augmentations

from . import Vid_dataset as vd


class InterlacedDataset(data.Dataset):
    '''
    Read interlaced and progressive frame triplets (pairs of three).
    Interlaced frame is expected to be "combed" from the progressive pair.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(InterlacedDataset, self).__init__()
        self.opt = opt
        self.paths_in, self.paths_top, self.paths_bot, self.paths_prog = None, None, None, None
        self.output_sample_imgs = None
        self.num_frames  = opt.get('num_frames', 3)
        
        # _, self.paths_in = util.get_image_paths('img', opt['dataroot_in'])
        # _, self.paths_top = util.get_image_paths('img', opt['dataroot_top'])
        # _, self.paths_bot = util.get_image_paths('img', opt['dataroot_bottom'])
        _, self.paths_prog = util.get_image_paths('img', opt['dataroot_progressive'])

        # self.paths_prog = opt.get('dataroot_progressive', None)
        # if self.paths_prog:
        #     self.video_list = os.listdir(self.paths_prog)

        # if self.paths_in and self.paths_top and self.paths_bot:
        #     assert len(self.paths_top) >= len(self.paths_in), \
        #         'Top dataset contains fewer images than interlaced dataset  - {}, {}.'.format(
        #         len(self.paths_top), len(self.paths_in))
        #     assert len(self.paths_bot) >= len(self.paths_in), \
        #         'Bottom dataset contains fewer images than interlaced dataset  - {}, {}.'.format(
        #         len(self.paths_bot), len(self.paths_in))

    def __getitem__(self, index):
        in_path, top_path, bot_path = None, None, None
        patch_size = self.opt['HR_size']
        idx_center = (self.num_frames - 1) // 2

        image_channels = self.opt.get('image_channels', 3)

        img_list = []

        if self.opt.get('combed', None):
            if index+self.num_frames*2 > len(self.paths_prog):
                index = index - self.num_frames*2

            interlaced_list = []
            for i_frame in range(self.num_frames * 2):
                img = util.read_img(None, self.paths_prog[int(index)+(i_frame)], out_nc=image_channels)
                img_list.append(img)
                
            if self.opt.get('frame_duping', None):
                if random.random() < 0.25:
                    idx = random.randint(1, (self.num_frames * 2) - 1)
                    img_list[idx-1] = img_list[idx]

            
            odds = img_list[0::2]
            evens = img_list[1::2]
            for i in range(self.num_frames):
                interlaced = np.zeros_like(odds[0])
                interlaced[i%2::2, :, :] = odds[i][i%2::2, :, :]
                interlaced[(i+1)%2::2, :, :] = evens[i][(i+1)%2::2, :, :]
                interlaced_list.append(interlaced)

            center_odd = odds[idx_center]
            center_even = evens[idx_center]

            t = self.num_frames
            HR_ODD = [np.asarray(GT) for GT in odds]  # list -> numpy # input: list (contatin numpy: [H,W,C])
            HR_EVEN = [np.asarray(GT) for GT in evens]  # list -> numpy # input: list (contatin numpy: [H,W,C])
            HR_ODD = np.asarray(HR_ODD) # numpy, [T,H,W,C]
            HR_EVEN = np.asarray(HR_EVEN) # numpy, [T,H,W,C]
            h_HR, w_HR, c = center_odd.shape #HR_center.shape #TODO: check, may be risky
            HR_ODD = HR_ODD.transpose(1,2,3,0).reshape(h_HR, w_HR, -1) # numpy, [H',W',CT]
            HR_EVEN = HR_EVEN.transpose(1,2,3,0).reshape(h_HR, w_HR, -1) # numpy, [H',W',CT]
            LR = [np.asarray(LT) for LT in interlaced_list]  # list -> numpy # input: list (contatin numpy: [H,W,C])
            LR = np.asarray(LR) # numpy, [T,H,W,C]
            LR = LR.transpose(1,2,3,0).reshape(h_HR, w_HR, -1) # numpy, [Hl',Wl',CT]

            if self.opt['phase'] == 'train':
                HR_ODD, LR, hr_crop_params, _ = vd.random_crop_mod(HR_ODD, LR, patch_size, 1)
                HR_EVEN, _ = vd.apply_crop_params(HR=HR_EVEN, hr_crop_params=hr_crop_params)

                HR_ODD = util.np2tensor(HR_ODD, bgr2rgb=True, add_batch=False) # Tensor, [CT',H',W'] or [T, H, W]
                HR_EVEN = util.np2tensor(HR_EVEN, bgr2rgb=True, add_batch=False) # Tensor, [CT',H',W'] or [T, H, W]
                LR = util.np2tensor(LR, bgr2rgb=True, add_batch=False) # Tensor, [CT',H',W'] or [T, H, W]

                HR_ODD = HR_ODD.view(c,t,patch_size,patch_size) # Tensor, [C,T,H,W]
                HR_EVEN = HR_EVEN.view(c,t,patch_size,patch_size) # Tensor, [C,T,H,W]
                LR = LR.view(c,t,patch_size,patch_size) # Tensor, [C,T,H,W]

                HR_ODD = HR_ODD.transpose(0,1) # Tensor, [T,C,H,W]
                HR_EVEN = HR_EVEN.transpose(0,1) # Tensor, [T,C,H,W]
                LR = LR.transpose(0,1) # Tensor, [T,C,H,W]
        else:
            if index + self.num_frames > len(self.paths_prog):
                index = index - self.num_frames

            for i_frame in range(self.num_frames):
                img = util.read_img(None, self.paths_prog[int(index)+(i_frame)], out_nc=image_channels)
                img_list.append(img)

            if self.opt.get('frame_duping', None):
                if random.random() < 0.25:
                    idx = random.randint(1, (self.num_frames) - 1)
                    img_list[idx-1] = img_list[idx]
                if random.random() < 0.025:  
                    frame = img_list[0]
                    for i in range(len(img_list)):
                        img_list[i] = frame

            HR = [np.asarray(GT) for GT in img_list]
            HR = np.asarray(HR)
            h_HR, w_HR, c = img_list[idx_center].shape
            HR = HR.transpose(1,2,3,0).reshape(h_HR, w_HR, -1)

            offset = 0
            if random.random() < 0.5:
                offset = 1
            for i in range(self.num_frames):
                h, w, c = img_list[i].shape
                if (i+offset)%2 == 0:
                    # Grab only the "odd" field and stretch it back to full height
                    img_list[i] = cv2.resize(img_list[i][0::2, :, :], (w, h), interpolation=cv2.INTER_NEAREST)
                else:
                    # Grab only the "even" field and stretch it back to full height
                    img_list[i] = cv2.resize(img_list[i][1::2, :, :], (w, h), interpolation=cv2.INTER_NEAREST)
                    # Shift stretched field down by one pixel to align it
                    img_list[i][1:, ...] = img_list[i][:-1, ...]

            LR = [np.asarray(LT) for LT in img_list]  # list -> numpy # input: list (contatin numpy: [H,W,C])
            LR = np.asarray(LR) # numpy, [T,H,W,C]
            LR = LR.transpose(1,2,3,0).reshape(h_HR, w_HR, -1) # numpy, [Hl',Wl',CT]

            HR, LR, hr_crop_params, _ = vd.random_crop_mod(HR, LR, patch_size, 1)
            # h_start_hr, h_end_hr, w_start_hr, w_end_hr = hr_crop_params
            # LR, _, _, _ = vd.random_crop_mod(LR, None, patch_size, 1, (h_start_hr//2, h_end_hr//2, w_start_hr, w_end_hr))

            HR = util.np2tensor(HR, bgr2rgb=True, add_batch=False) # Tensor, [CT',H',W'] or [T, H, W]
            LR = util.np2tensor(LR, bgr2rgb=True, add_batch=False) # Tensor, [CT',H',W'] or [T, H, W]

            HR = HR.view(c,self.num_frames,patch_size,patch_size) # Tensor, [C,T,H,W]
            LR = LR.view(c,self.num_frames,patch_size,patch_size) # Tensor, [C,T,H,W]

            HR = HR.transpose(0,1) # Tensor, [T,C,H,W]
            LR = LR.transpose(0,1) # Tensor, [T,C,H,W]
        
        # return {'LR': LR, 'HR_ODD': HR_ODD, 'HR_EVEN': HR_EVEN, 'HR': HR_ODD, 'HR_center': HR_ODD[idx_center, :, :, :], 'LR_bicubic': []}
        return {'LR': LR, 'HR': HR, 'HR_center': HR[idx_center, :, :, :], 'LR_bicubic': []}

    def __len__(self):
        return len(self.paths_prog)


class InterlacedTestDataset(data.Dataset):
    '''Read interlaced images only in the test phase.'''

    def __init__(self, opt):
        super(InterlacedTestDataset, self).__init__()
        self.opt = opt
        self.paths_in = None

        self.num_frames = opt['num_frames']

        # read image list from lmdb or image files
        _, self.paths_in = util.get_image_paths('img', opt['dataroot_in'])
        assert self.paths_in, 'Error: Interlaced paths are empty.'

    def __getitem__(self, index):
        in_path = None

        # # get LR image
        # in_path = self.paths_in[index]
        # #img_LR = util.read_img(self.LR_env, LR_path)
        # img_in = util.read_img(None, in_path)

        # # BGR to RGB, HWC to CHW, numpy to tensor
        # img_in = util.np2tensor(img_in, add_batch=False)

        paths_in = self.paths_in

        for i_frame in range(self.num_frames):
            if index == len(self.paths_in)-2 and self.num_frames == 3:
                # print("second to last frame:", i_frame)
                if i_frame == 0:
                    LR_img = util.read_img(None, paths_in[int(index)], out_nc=self.image_channels)
                else:
                    LR_img = util.read_img(None, paths_in[int(index)+1], out_nc=self.image_channels)
            elif index == len(self.paths_in)-1 and self.num_frames == 3:
                # print("last frame:", i_frame)
                LR_img = util.read_img(None, paths_in[int(index)], out_nc=self.image_channels)
            # every other internal frame
            else:
                # print("normal frame:", idx_frame)
                LR_img = util.read_img(None, paths_in[int(index)+(i_frame)], out_nc=self.image_channels)
            #TODO: check if this is necessary
            LR_img = util.modcrop(LR_img, scale)
    

            LR_list.append(LR_img) # h, w, c
            
            if not self.y_only and (not h_LR or not w_LR):
                h_LR, w_LR, c = LR_img.shape

        t = self.num_frames
        LR = [np.asarray(LT) for LT in LR_list]  # list -> numpy # input: list (contatin numpy: [H,W,C])
        LR = np.asarray(LR) # numpy, [T,H,W,C]
        LR = LR.transpose(1,2,3,0).reshape(h_LR, w_LR, -1) # numpy, [Hl',Wl',CT]

        LR = util.np2tensor(LR, normalize=znorm, bgr2rgb=True, add_batch=False) # Tensor, [CT',H',W'] or [T, H, W]
        LR = LR.view(c,t,h_LR,w_LR) # Tensor, [C,T,H,W]
        if self.shape == 'TCHW':
            LR = LR.transpose(0,1) # Tensor, [T,C,H,W]

        return {'LR': LR, 'lr_path': in_path}

    def __len__(self):
        return len(self.paths_in)
