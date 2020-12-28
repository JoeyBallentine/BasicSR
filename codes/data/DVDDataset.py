import os
import uuid

import cv2
import torch.utils.data as data

import codes.dataops.augmentations as augmentations
import codes.dataops.common as util


class DVDDataset(data.Dataset):
    """
    Read interlaced and progressive frame triplets (pairs of three).
    Interlaced frame is expected to be "combed" from the progressive pair.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super(DVDDataset, self).__init__()
        self.opt = opt
        self.paths_in, self.paths_top, self.paths_bot, self.paths_progressive = None, None, None, None
        self.output_sample_images = None  # DEBUG use only! Should be a path to save debug images, None to disable

        _, self.paths_in = util.get_image_paths('img', opt['dataroot_in'])
        _, self.paths_top = util.get_image_paths('img', opt['dataroot_top'])
        _, self.paths_bot = util.get_image_paths('img', opt['dataroot_bottom'])
        _, self.paths_progressive = util.get_image_paths('img', opt['dataroot_progressive'])

        if self.paths_in:
            if self.paths_top:
                if len(self.paths_top) < len(self.paths_in):
                    raise ValueError(
                        'Top dataset contains less images than interlaced dataset  - {}, {}.'.format(
                            len(self.paths_top), len(self.paths_in)
                        )
                    )
            if self.paths_bot:
                if len(self.paths_bot) < len(self.paths_in):
                    raise ValueError(
                        'Bottom dataset contains less images than interlaced dataset  - {}, {}.'.format(
                            len(self.paths_bot), len(self.paths_in)
                        )
                    )

    def __getitem__(self, index):
        in_path, top_path, bot_path = None, None, None
        patch_size = self.opt['HR_size']

        image_channels = self.opt.get('image_channels', 3)

        if self.paths_progressive:
            has_next = len(self) != index + 1
            top_path = self.paths_progressive[index - 0 if has_next else 1]
            bot_path = self.paths_progressive[index + 1 if has_next else 0]
        else:
            top_path = self.paths_top[index]
            bot_path = self.paths_bot[index]

        img_top = util.read_img(None, top_path, out_nc=image_channels)
        img_bot = util.read_img(None, bot_path, out_nc=image_channels)

        # Read interlaced frame or create interlaced image from top/bottom frames
        if self.paths_in is None:
            img_in = img_top.copy()
            img_in[1::2, :, :] = img_bot[1::2, :, :]
        else:
            in_path = self.paths_in[index]
            img_in = util.read_img(None, in_path, out_nc=image_channels)

        if self.opt['phase'] == 'train':
            # Random Crop (reduce computing cost and adjust images to correct size first)
            for img_hr in img_top, img_bot:
                if img_hr.shape[0] > patch_size or img_hr.shape[1] > patch_size:
                    img_top, img_bot, img_in = augmentations.random_crop_dvd(img_top, img_bot, img_in, patch_size)

        # Debug
        # TODO: do not leave on during real training
        # TODO: use the debugging functions to visualize or save images instead
        # Save img_in, img_top, and img_bot images to a directory to visualize what is the result of the on the fly
        # augmentations.
        if self.opt['phase'] == 'train' and self.output_sample_images:
            _, im_name = os.path.split(top_path)
            debug_path = os.path.join(self.output_sample_images, 'Sample_OTF_Images')
            if not os.path.exists(debug_path):
                os.makedirs(debug_path)
            rand_uuid = uuid.uuid4().hex
            debug_path = os.path.join(debug_path, rand_uuid + '_' + im_name)
            for name, var in [('interlaced', img_in), ('top', img_top), ('bottom', img_bot)]:
                cv2.imwrite(debug_path + '_' + name + '.png', var)

        img_in = util.np2tensor(img_in, add_batch=False)
        img_top = util.np2tensor(img_top, add_batch=False)
        img_bot = util.np2tensor(img_bot, add_batch=False)

        return {
            'in': img_in, 'in_path': in_path or 'OTF',
            'top': img_top, 'top_path': top_path,
            'bottom': img_bot, 'bot_path': bot_path
        }

    def __len__(self):
        return len(self.paths_top or self.paths_progressive)
