import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']

    if model == 'sr':
        from .SR_model import SRModel as M
    elif model == 'srgan' or model == 'srragan' or model == 'srragan_hfen' or model == 'lpips':
        from .SRRaGAN_model import SRRaGANModel as M
    elif model == 'sftgan':
        from .SFTGAN_ACD_model import SFTGAN_ACD_Model as M
    # elif model == 'srragan_n2n':
        # from .SRRaGAN_n2n_model import SRRaGANModel as M
    elif model == 'ppon':
        from .ppon_model import PPONModel as M
    elif model == 'asrragan':
        from .ASRRaGAN_model import ASRRaGANModel as M
    elif model == 'vsrgan':
        from .VSR_model import VSRModel as M
    elif model == 'pbr':
        from .PBR_model import PBRModel as M
    elif model == 'dvd':
        from .DVD_model import DVDModel as M
    elif model == 'vdof':
        from .VDOF2_model import VDOF2Model as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
