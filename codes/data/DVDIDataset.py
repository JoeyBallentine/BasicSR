import torch.utils.data as data

import codes.dataops.common as util


class DVDIDataset(data.Dataset):
    """Read interlaced images only in the test phase."""

    def __init__(self, opt):
        super(DVDIDataset, self).__init__()
        self.opt = opt
        _, self.paths_in = util.get_image_paths('img', opt['dataroot_in'])
        if not self.paths_in:
            raise ValueError("No images in interlaced path.")

    def __getitem__(self, index):
        # get LR image
        in_path = self.paths_in[index]
        # img_LR = util.read_img(self.LR_env, LR_path)
        img_in = util.read_img(None, in_path)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_in = util.np2tensor(img_in, add_batch=False)
        return {'in': img_in, 'in_path': in_path}

    def __len__(self):
        return len(self.paths_in)
