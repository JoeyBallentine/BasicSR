import logging
import os
import os.path as osp
from collections import OrderedDict

import cv2

# PAD_MOD
_str_to_cv2_pad_to = {
    'constant': cv2.BORDER_CONSTANT,
    'edge': cv2.BORDER_REPLICATE,
    'reflect': cv2.BORDER_REFLECT_101,
    'symmetric': cv2.BORDER_REFLECT
}
# INTER_MODE
_str_to_cv2_interpolation = {
    'nearest': cv2.INTER_NEAREST,
    'linear': cv2.INTER_LINEAR,
    'bilinear': cv2.INTER_LINEAR,
    'area': cv2.INTER_AREA,
    'cubic': cv2.INTER_CUBIC,
    'bicubic': cv2.INTER_CUBIC,
    'lanczos': cv2.INTER_LANCZOS4,
    'lanczos4': cv2.INTER_LANCZOS4,
    'linear_exact': cv2.INTER_LINEAR_EXACT,
    'matlab_linear': 773,
    'matlab_box': 774,
    'matlab_lanczos2': 775,
    'matlab_lanczos3': 776,
    'matlab_bicubic': 777,
    'realistic': 999
}


def parse2lists(types):
    """Converts dictionaries or single string options to lists that work with random choice"""
    if isinstance(types, dict):
        types_list = []
        for k, v in types.items():
            types_list.extend([k] * v)
        types = types_list
    elif isinstance(types, str):
        types = [types]
    # else:
    #     raise TypeError("Unrecognized blur type, must be list, dict or a string")

    # if(isinstance(types, list)):
    #     pass

    return types


def parse(opt_path: str, is_train: bool = True) -> dict:
    """
    Parse options file.
    :param opt_path: option file path. supports JSON and YAML.
    :param is_train: indicate whether in training or not. default: True.
    :returns: parsed options
    """

    # todo ; perhaps replace is_train with `phase: str = 'train'`?

    # check if configuration file exists
    if not os.path.isfile(opt_path):
        opt_path = os.path.join('options', 'train' if is_train else 'test', opt_path)
        if not os.path.isfile(opt_path):
            print('Configuration file "{:s}" not found.'.format(opt_path))
            exit(1)

    ext = osp.splitext(opt_path)[1].lower()
    if ext == '.json':
        import json
        # remove comments starting with '//'
        # todo ; replace JSON support with YAML
        #        comments on JSON is not official or part of spec, so removing everything after // is risky
        #        as it might be in a file path or such, resulting in bad data. It's simply too risky.
        #        YAML on the other hand supports comments natively and is MUCH MUCH more human readable.
        json_str = ''
        with open(opt_path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.split('//')[0] + '\n'
                json_str += line
        opt = json.loads(json_str)
    elif ext in ['.yml', '.yaml']:
        import yaml
        import re
        try:
            from yaml import CLoader as Loader  # C loader, faster
        except ImportError:
            from yaml import Loader  # Python loader, slower
        Loader.add_constructor(
            tag=yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            constructor=lambda loader, node: OrderedDict(loader.construct_pairs(node))
        )
        # compiled resolver to correctly parse scientific notation numbers
        Loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+]?[0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        with open(opt_path, 'rt', encoding='utf-8') as f:
            opt = yaml.load(f, Loader=Loader)

    opt['is_train'] = is_train
    scale = opt.get('scale', 4)
    bm = opt.get('batch_multiplier', None)

    # datasets
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        dataset['scale'] = scale
        is_lmdb = False
        if dataset.get('dataroot_HR', None) is not None:
            hr_images_paths = dataset['dataroot_HR']
            if type(hr_images_paths) is list:
                # todo ; needs is_lmdb support
                dataset['dataroot_HR'] = [os.path.expanduser(x) for x in hr_images_paths]
            elif type(hr_images_paths) is str:
                dataset['dataroot_HR'] = os.path.expanduser(hr_images_paths)
                is_lmdb = dataset['dataroot_HR'].endswith('lmdb')
        if dataset.get('dataroot_HR_bg', None) is not None:
            hr_images_paths = dataset['dataroot_HR_bg']
            if type(hr_images_paths) is list:
                dataset['dataroot_HR_bg'] = [os.path.expanduser(x) for x in hr_images_paths]
            elif type(hr_images_paths) is str:
                dataset['dataroot_HR_bg'] = os.path.expanduser(hr_images_paths)
        if dataset.get('dataroot_LR', None) is not None:
            lr_images_paths = dataset['dataroot_LR']
            if type(lr_images_paths) is list:
                # todo ; needs is_lmdb support
                dataset['dataroot_LR'] = [os.path.expanduser(x) for x in lr_images_paths]
            elif type(lr_images_paths) is str:
                dataset['dataroot_LR'] = os.path.expanduser(lr_images_paths)
                is_lmdb = dataset['dataroot_LR'].endswith('lmdb')
        dataset['data_type'] = 'lmdb' if is_lmdb else 'img'

        if phase == 'train' and bm:
            dataset['virtual_batch_size'] = bm * dataset["batch_size"]

        if phase == 'train' and dataset.get('subset_file', None) is not None:
            dataset['subset_file'] = os.path.expanduser(dataset['subset_file'])

        if dataset.get('lr_downscale_types', None) is not None:
            if not isinstance(dataset['lr_downscale_types'], list):
                dataset['lr_downscale_types'] = [dataset['lr_downscale_types']]
            downscale_types = []
            for algo in dataset['lr_downscale_types']:
                if isinstance(algo, str):
                    algo = _str_to_cv2_interpolation[algo.lower()]
                downscale_types.append(algo)
            dataset['lr_downscale_types'] = downscale_types

        for t, dep in [
            ('lr_blur_types', 'lr_blur'), ('lr_noise_types', 'lr_noise'),
            ('lr_noise_types2', 'lr_noise2'), ('hr_noise_types', 'hr_noise')
        ]:
            if dataset.get(t, None) and dataset.get(dep, None):
                dataset[t] = parse2lists(dataset[t])

        tensor_shape = dataset.get('tensor_shape', None)
        if tensor_shape:
            opt['tensor_shape'] = tensor_shape

    # path
    opt['path'] = {k: os.path.expanduser(v) for k, v in opt['path'].items()}
    if is_train:
        opt['path']['experiments_root'] = os.path.join(opt['path']['root'], 'experiments', opt['name'])
        opt['path']['models'] = os.path.join(opt['path']['experiments_root'], 'models')
        opt['path']['training_state'] = os.path.join(opt['path']['experiments_root'], 'training_state')
        opt['path']['log'] = opt['path']['experiments_root']
        opt['path']['val_images'] = os.path.join(opt['path']['experiments_root'], 'val_images')
        opt['train']['overwrite_val_imgs'] = opt['train'].get('overwrite_val_imgs', None)
        opt['train']['val_comparison'] = opt['train'].get('val_comparison', None)
        opt['logger']['overwrite_chkp'] = opt['logger'].get('overwrite_chkp', None)
        fsa = opt['train'].get('use_frequency_separation', None)
        if fsa and not opt['train'].get('fs', None):
            opt['train']['fs'] = fsa
        # change some options for debug mode
        if 'debug_nochkp' in opt['name']:
            opt['train']['val_freq'] = 8
            opt['logger']['print_freq'] = 2
            opt['logger']['save_checkpoint_freq'] = 1000  # 10000000
            opt['train']['lr_decay_iter'] = 10
        elif 'debug' in opt['name']:
            opt['train']['val_freq'] = 8
            opt['logger']['print_freq'] = 2
            opt['logger']['save_checkpoint_freq'] = 8
            opt['train']['lr_decay_iter'] = 10
    else:
        opt['path']['results_root'] = os.path.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['log'] = opt['path']['results_root']

    # network
    opt['network_G']['scale'] = scale

    # relative learning rate and options
    if 'train' in opt:
        niter = opt['train']['niter']
        for o in ["T_period", "restarts", "lr_steps", "lr_steps_inverse", "swa_start_iter"]:
            if o == "swa_start_iter":
                opt['train'][o] = int(opt['train'][o + '_rel'] * niter)
            else:
                opt['train'][o] = [int(x * niter) for x in opt['train'][o + '_rel']]
            opt['train'].pop(o + '_rel')

    # export CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def dict2str(opt, indent_l=1):
    """dict to string for logger"""
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


def check_resume(opt):
    """Check resume states and pretrain_model paths"""
    logger = logging.getLogger('base')
    if opt['path']['resume_state']:
        if opt['path']['pretrain_model_G'] or opt['path']['pretrain_model_D']:
            logger.warning('pretrain_model path will be ignored when resuming training.')

        state_idx = osp.basename(opt['path']['resume_state']).split('.')[0]
        opt['path']['pretrain_model_G'] = osp.join(opt['path']['models'], '{}_G.pth'.format(state_idx))
        logger.info('Set [pretrain_model_G] to ' + opt['path']['pretrain_model_G'])
        if 'gan' in opt['model']:
            opt['path']['pretrain_model_D'] = osp.join(opt['path']['models'], '{}_D.pth'.format(state_idx))
            logger.info('Set [pretrain_model_D] to ' + opt['path']['pretrain_model_D'])
        if 'swa' in opt['model'] or opt['swa']:
            opt['path']['pretrain_model_swaG'] = osp.join(opt['path']['models'], '{}_swaG.pth'.format(state_idx))
            logger.info('Set [pretrain_model_swaG] to ' + opt['path']['pretrain_model_swaG'])
