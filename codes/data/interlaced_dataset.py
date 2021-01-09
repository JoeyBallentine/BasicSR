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

        # idx_video = random.randint(0, len(self.video_list)-1)
        # video_dir = self.video_list[idx_video]

        # _, paths_prog = util.get_image_paths(self.opt['data_type'], path.join(self.paths_prog, video_dir))

        # # random reverse augmentation
        # random_reverse = self.opt.get('random_reverse', False)
        
        # # skipping intermediate frames to learn from low FPS videos augmentation
        # # testing random frameskip up to 'max_frameskip' frames
        # max_frameskip = self.opt.get('max_frameskip', 0)
        # if max_frameskip > 0:
        #     max_frameskip = min(max_frameskip, len(paths_prog)//(self.num_frames-1))
        #     frameskip = random.randint(1, max_frameskip)
        # else:
        #     frameskip = 1
        # # print("max_frameskip: ", max_frameskip)

        # assert ((self.num_frames-1)*frameskip) <= (len(paths_prog)-1), (
        #     f'num_frame*frameskip must be smaller than the number of frames per video, check {video_dir}')
        
        # # if number of frames of training video is for example 31, "max index -num_frames" = 31-3=28
        # idx_frame = random.randint(0, (len(paths_HR)-1)-((self.num_frames-1)*frameskip))
        # # print('frameskip:', frameskip)

        # adjust index if would load out of range
        if index+self.num_frames*2 > len(self.paths_prog):
            index = index - self.num_frames*2

        for i_frame in range(self.num_frames * 2):
            img = util.read_img(None, self.paths_prog[int(index)+(i_frame)], out_nc=image_channels)
            img_list.append(img)

        # if self.paths_prog:
        #     if index+1 != len(self):
        #         top_path = self.paths_prog[index]
        #         bot_path = self.paths_prog[index+1]
        #     else:
        #         top_path = self.paths_prog[index-1]
        #         bot_path = self.paths_prog[index]
        # else:
        #     top_path = self.paths_top[index]
        #     bot_path = self.paths_bot[index]

        # img_top = util.read_img(None, top_path, out_nc=image_channels)
        # img_bot = util.read_img(None, bot_path, out_nc=image_channels)

        interlaced_list = []

        if self.opt.get('frame_duping', None):
            if random.random() < 0.25:
                idx = random.randint(1, (self.num_frames * 2) - 1)
                print(idx, len(img_list))
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


        # Read interlaced frame or create interlaced image from top/bottom frames
        # if self.paths_in is None:
        #     img_in = img_top.copy()
        #     img_in[1::2, :, :] = img_bot[1::2, :, :]
        # else:
        #     in_path = self.paths_in[index]
        #     img_in = util.read_img(None, in_path, out_nc=image_channels)
        
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
            
        return {'LR': LR, 'HR_ODD': HR_ODD, 'HR_EVEN': HR_EVEN, 'HR': HR_ODD, 'HR_center': HR_ODD[idx_center, :, :, :], 'LR_bicubic': []}

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
