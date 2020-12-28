"""create dataset and dataloader"""
import logging

import torch.utils.data

from codes.data.DVDDataset import DVDDataset
from codes.data.DVDIDataset import DVDIDataset
from codes.data.LRHRC_dataset import LRHRDataset as LRHRCDataset
from codes.data.LRHROTF_dataset import LRHRDataset as LRHROTFDataset
from codes.data.LRHRPBR_dataset import LRHRDataset as LRHRPBRDataset
from codes.data.LRHR_dataset import LRHRDataset
from codes.data.LRHR_seg_bg_dataset import LRHRSeg_BG_Dataset
from codes.data.LR_dataset import LRDataset
from codes.data.Vid_dataset import VidTestsetLoader
from codes.data.Vid_dataset import VidTrainsetLoader


def create_dataloader(dataset: torch.utils.data.Dataset, dataset_opt: dict) -> torch.utils.data.DataLoader:
    """
    Create Dataloader
    :param dataset: PyTorch Dataset
    :param dataset_opt: Options to pass to the Dataloader
    """
    if dataset_opt['phase'] == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['n_workers'],
            drop_last=True,
            pin_memory=True
        )
    else:
        # todo ; what is this "else" phase, hardcode it
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )


def create_dataset(dataset_opt: dict) -> torch.utils.data.Dataset:
    """
    Create Dataset
    :param dataset_opt: Options to pass to the Dataset
    """
    mode = dataset_opt['mode']
    dataset = {
        "LR": LRDataset,
        "LRHR": LRHRDataset,
        "LRHROTF": LRHROTFDataset,
        "LRHRC": LRHRCDataset,
        "LRHRPBR": LRHRPBRDataset,
        "LRHRseg_bg": LRHRSeg_BG_Dataset,
        "VLRHR": VidTrainsetLoader,
        "VLR": VidTestsetLoader,
        "DVD": DVDDataset,
        "DVDI": DVDIDataset
    }.get(mode, None)
    if dataset is None:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = dataset(dataset_opt)
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
