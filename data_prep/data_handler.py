# from data_prep.loadData import Dataset
from data_prep.loadData_spireEMA import Dataset_prepare_dump, Dataset_prepare_dump_collate
from torch.utils.data import DataLoader
from common.utils import support_aditional_feats
import torch
from pathlib import Path
import os
import librosa
from tqdm import tqdm
from pqdm.threads import pqdm

def collect(cfg):
    collate_fn = Dataset_prepare_dump_collate(cfg.common.mode.feats, cfg.common.mode.pyannote_sub_embed, cfg=cfg)
    # if cfg.common.dump_feats:
    #     dump_feats(cfg)
    # support_aditional_feats(cfg)
    loaders = []
    for mode in ['train', 'val', 'test', 'unseen_spk_val', 'unseen_spk_test']:
        dataset = Dataset_prepare_dump(mode, cfg.common, cfg, **cfg.data )
        # print(dataset[0])
        loader_ = DataLoader(   
                            dataset, 
                            shuffle=True if mode == 'train' else False, 
                            batch_size=int(cfg.common.mode.batch_size), 
                            collate_fn=collate_fn, num_workers=4, pin_memory=True, 
                            )
        loaders.append(loader_)
    # global_stats(loaders[0])
    # exit()
    return loaders


# def global_stats(loader):
#     for data in loader:
#         ema_padded, mfcc_padded, mel_padded, phon_padded, dur_padded, ema_lens, mel_lens, phon_lens, spkids, spks, names, phs, tphn_padded = data

def get_files(path, extension='.wav'):
    if isinstance(path, str): path = Path(path).expanduser().resolve()
    return list(path.rglob(f'*{extension}'))

