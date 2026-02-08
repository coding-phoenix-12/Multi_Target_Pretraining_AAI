import os
from common.logger import logger
import scipy.io
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import librosa
from multiprocessing import Pool
from tqdm import tqdm
from librosa.filters import mel as librosa_mel_fn
from scipy.io import loadmat
import re
import random
import librosa

class Dataset_prepare_dump(torch.utils.data.Dataset):
    def __init__(self, mode, common, config, **kwargs):
        self.__dict__.update(kwargs)
        self.comm = common
        self.config = config
        self.subjects = common.sub
        self.mode = mode

        self.rootPath = self.config.data.rootPath
        # assert self.subjects != 'all' #test for predefined
        self.unseen_test_subjects = ['spk2', 'spk32', 'spk19', 'spk27', 'spk22', 'spk37']
        spk = None
        if mode.startswith('unseen'):
            spk = self.unseen_test_subjects
            self.subs = self.unseen_test_subjects
            # mode = mode.split('_')[-1]
        self.parse_file_based_on_mode(mode, spk)
        
        self.classes = torch.load("/data/jesuraj/AAI/pretraining/classes.pt", weights_only=False)
        self.phonemes = torch.load("/data/jesuraj/AAI/pretraining/phoneme_class.pt", weights_only=False)
        
        self.art_label_map = {}
        for cl in self.classes.keys():
            for art_ind, label in enumerate(self.classes[cl]):
                self.art_label_map[label] = art_ind + 1
                
                
        self.art_weights = {
                                "d": np.expand_dims(np.array([0, 0, 0, 0, 0, 0, 1 ,1 ,0, 0, 0, 0]), axis=1),
                                "f": np.expand_dims(np.array([0, 0, 1, 1, 0, 0, 0 ,0 ,0, 0, 0, 0]), axis=1),
                                "g": np.expand_dims(np.array([0, 0, 0, 0, 0, 0, 0 ,0 ,0, 0, 1, 1]), axis=1),
                                "k": np.expand_dims(np.array([0, 0, 0, 0, 0, 0, 0 ,0 ,0, 0, 1, 1]), axis=1),
                                "m": np.expand_dims(np.array([0, 0, 1, 1, 0, 0, 0 ,0 ,0, 0, 0, 0]), axis=1),
                                "n": np.expand_dims(np.array([0, 0, 0, 0, 0, 0, 1 ,1 ,0, 0, 0, 0]), axis=1),
                                "p": np.expand_dims(np.array([0, 0, 1, 1, 0, 0, 0 ,0 ,0, 0, 0, 0]), axis=1),
                                "s": np.expand_dims(np.array([0, 0, 0, 0, 0, 0, 1 ,1 ,0, 0, 1, 1]), axis=1),
                                "t": np.expand_dims(np.array([0, 0, 0, 0, 0, 0, 1 ,1 ,0, 0, 0, 0]), axis=1),
                                "v": np.expand_dims(np.array([0, 0, 1, 1, 0, 0, 0 ,0 ,0, 0, 0, 0]), axis=1),
                                "z": np.expand_dims(np.array([0, 0, 0, 0, 0, 0, 1 ,1 ,0, 0, 1, 1]), axis=1),
                                "th": np.expand_dims(np.array([0, 0, 0, 0, 0, 0, 1 ,1 ,0, 0, 0, 0]), axis=1),
                                "dh": np.expand_dims(np.array([0, 0, 0, 0, 0, 0, 1 ,1 ,0, 0, 0, 0]), axis=1),
                                
                        }
            
        
        
        self.return_keys = [
            'mfcc', 
            'phone_align', 
            'phone_ids_align', 
            'phones', 
            'phone_ids'
        ]  
          
          
        
        
    def parse_file_based_on_mode(self, mode, speakers=None):
        files = []
        self.files = []
       
        if "unseen" in mode:
            spire_ema_files = sorted([x for x in os.listdir(os.path.join(self.rootPath, 'tmp')) if "SpireEMA" in x and x.split("_")[1] in self.unseen_test_subjects])
            
        else:
            spire_ema_files = sorted([x for x in os.listdir(os.path.join(self.rootPath, 'tmp')) if "SpireEMA" in x and x.split("_")[1] not in self.unseen_test_subjects])
            if "train" in mode and "ema" not in self.config.common.predict:
                if "libri360" in self.config.common.datasetNames:
                    self.files += [os.path.join('/home2/data/jesuraj/AAI/data/tmp/', x) for x in os.listdir(os.path.join('/home2/data/jesuraj/AAI/data/tmp/'))]

                if "libri100" in self.config.common.datasetNames:
                    self.files += [os.path.join(self.rootPath, 'tmp', x) for x in os.listdir(os.path.join(self.rootPath, 'tmp')) if "SpireEMA" not in x]
        
        
        spks = list(set([x.split("_")[1] for x in spire_ema_files]))
        total_train_files = 0
        used_train_files = 0
        pick_odd_even = False
        for spk_idx, sub in enumerate(spks):
            for idx, ptfile in enumerate(sorted([x for x in spire_ema_files if x.split("_")[1] == sub])):
                if ((idx + 10) % 10)==0:
                    if 'test' in mode: 
                        files.append(os.path.join(self.rootPath, 'tmp', ptfile))
                elif ((idx+10-1)%10)==0:
                    if 'val' in mode: 
                        files.append(os.path.join(self.rootPath, 'tmp',ptfile))
                else:
                    if 'train' in mode and "SpireEMA" in self.config.common.datasetNames:
                        total_train_files += 1
                        if idx % 10 not in self.config.common.exclude_utts:
                            if self.config.common.exclude_odd:
                                if pick_odd_even:
                                    files.append(os.path.join(self.rootPath, 'tmp',ptfile))
                                    used_train_files += 1
                                    pick_odd_even = False
                                else:
                                    pick_odd_even = True
                            else:
                                files.append(os.path.join(self.rootPath, 'tmp',ptfile))
                                used_train_files += 1
        # available_files = {Path(f).stem:f for f in get_files(self.dumpdir, '.pt')}
        
        self.files += [f for f in files]
        
        
        
        
        logger.info(f'Folder path {self.files[0]}')
        logger.info(f'{len(self.files)} data being used in {mode}')
        if mode == "train":
            logger.info(f'Percentage used {used_train_files / total_train_files * 100}')
        # logger.info(f'spks {list(set([x.split("/")[-1].split("_")[1] for x in self.files]))}')
    
    
    
    def __len__(self):
        return len(self.files)     
            
    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        filename = self.files[idx]
        # print(filename)
        
        min_len = min(len(data["phone_align"]), len(data["mfcc"]))
        
        data["mfcc"] = data["mfcc"][:min_len, :]
        data["phone_align"] = data["phone_align"][:min_len]
        data["phone_ids_align"] = np.array(data["phone_ids_align"][:min_len])
        data["phones"] = data["phones"][:min_len]
        # print(data["phone_ids"])
        data["phone_ids"] = np.array(data["phone_ids"][:min_len])
        
        
        if "ema" in self.config.common.predict:
            ema_norm = data["ema_norm"][: min_len, :]
            # ema_norm = data["ema_clipped"][: min_len, :]
            # ema_norm = np.delete(ema_norm, [4,5,6,7,10,11],1)
            # ema_norm = (ema_norm - np.mean(ema_norm, axis=0, keepdims=True)) / np.std(ema_norm, axis=0, keepdims=True)
            

            
        else:
            ema_norm = None

        if self.config.common.inputs == "feats":
            feats = torch.load(os.path.join(self.config.data.featsPath, filename.split("/")[-1]), weights_only=False)[: min_len, :]
        else:
            feats = None
            
        aligned_phones = data["phone_align"]
        unaligned_phones = data["phones"]
        
        
        # data["phone_ids_align"] = (data["phone_ids_align"]) 
        # data["phone_ids"] = (data["phone_ids"])
        
        art_labels = self.articulatory_labels(aligned_phones)
        unaligned_art_labels = self.articulatory_labels(unaligned_phones)
   
        
        
        art_weights, weight_mask = self.articulatory_weights(aligned_phones)
        
        
        

        
        data = [data[d] for d in self.return_keys]
        data.append(filename)
        data.append(art_labels)
        # data.append([None])
        
       
        data.append(ema_norm)
        data.append(art_weights)
 
        data.append(weight_mask)
        data.append(unaligned_art_labels)
        data.append(len(unaligned_phones))
        data.append(feats)
        
        
        return data
        
        
        
        
    def articulatory_weights(self, aligned_phones):
        weights = []
        mask = []
        for ph in aligned_phones:
            
            if ph in self.art_weights.keys():
                weights.append(self.art_weights[ph])
                mask.append(1)
            else:
                weights.append(np.zeros((12, 1)))
                mask.append(0)
                
        return np.array(weights).squeeze(-1), np.array(mask)
            
        
        
        
    def articulatory_labels(self, phones):
        place_label = []
        manner_label = []
        height_label = []
        backness_label = []
        
        # print(phones)
        for ph in phones:
            place_label.append(self.art_label_map[self.phonemes[ph][0]])
            manner_label.append(self.art_label_map[self.phonemes[ph][1]])
            height_label.append(self.art_label_map[self.phonemes[ph][2]])
            backness_label.append(self.art_label_map[self.phonemes[ph][3]])
            
            
        place_label, manner_label, height_label, backness_label = np.array(place_label), np.array(manner_label), np.array(height_label), np.array(backness_label)
        return np.stack([place_label, manner_label, height_label, backness_label]).T
                
            
            
        
    
    
class Dataset_prepare_dump_collate():
    
    def __init__(self, feat=False, subembed=False, cfg=None):
        self.feat = True if feat != 'mfcc' else False
        self.subembed = subembed
        self.cfg = cfg

    def __call__(self, batch):
        #name, ema, lens, mfcc, mfcclen, mel, mellen, ph, phid, dur, spkid, spk
        ema_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_ema_len = ema_lengths[0]


        max_mfcc_len = max([x[0].shape[0] for x in batch])
        max_phone_len = max([x[11] for x in batch])
        
        
        ema_padded = torch.FloatTensor(len(batch), max_ema_len, 12)
        ema_padded.zero_()

        mfcc_padded = torch.FloatTensor(len(batch), max_mfcc_len, 13)
        mfcc_padded.zero_()
        
        art_labels_padded = torch.FloatTensor(len(batch), max_ema_len, 4)
        art_labels_padded.zero_()
        
        phnid_align_padded = torch.LongTensor(len(batch), max_mfcc_len)
        phnid_align_padded.zero_()
        
        art_weights_padded = torch.FloatTensor(len(batch), max_mfcc_len, 12)
        art_weights_padded.zero_()
        
        weights_mask_padded = torch.FloatTensor(len(batch), max_mfcc_len, 1)
        weights_mask_padded.zero_()
        
        unaligned_art_labels = torch.FloatTensor(len(batch), max_phone_len, 4)
        unaligned_art_labels.zero_()
        
        phnid_padded = torch.LongTensor(len(batch), max_phone_len)
        phnid_padded.zero_()

        feats_padded = torch.FloatTensor(len(batch), max_mfcc_len, 768)
        feats_padded.zero_()
        
        
        ema_lens, mfcc_lens, names, phone_lens = [], [], [], []
        for i in range(len(ids_sorted_decreasing)):
            names.append(batch[ids_sorted_decreasing[i]][5])
            # ema_padded[i, :len(batch[ids_sorted_decreasing[i]][2]), :] = torch.from_numpy(batch[ids_sorted_decreasing[i]][2])
            
            art_labels_padded[i, :len(batch[ids_sorted_decreasing[i]][6]), :] = torch.from_numpy(batch[ids_sorted_decreasing[i]][6])
            mfcc_padded[i, :len(batch[ids_sorted_decreasing[i]][0]), :] = torch.from_numpy(batch[ids_sorted_decreasing[i]][0])
            if "ema" in self.cfg.common.predict:
                ema_padded[i, :len(batch[ids_sorted_decreasing[i]][7]), :] = torch.from_numpy(batch[ids_sorted_decreasing[i]][7])
                
            if self.cfg.common.inputs == "feats":
                # print(batch[ids_sorted_decreasing[i]][12].shape)
                feats_padded[i, :len(batch[ids_sorted_decreasing[i]][12]), :] = torch.from_numpy(batch[ids_sorted_decreasing[i]][12])
    
            phnid_align_padded[i, :len(batch[ids_sorted_decreasing[i]][2])] = torch.from_numpy(batch[ids_sorted_decreasing[i]][2])
    
            art_weights_padded[i, :len(batch[ids_sorted_decreasing[i]][8]), :] = torch.from_numpy(batch[ids_sorted_decreasing[i]][8])
            weights_mask_padded[i, :len(batch[ids_sorted_decreasing[i]][8]), 0] = torch.from_numpy(batch[ids_sorted_decreasing[i]][9])
    
            ema_lens.append(len(batch[ids_sorted_decreasing[i]][0]))
            mfcc_lens.append(len(batch[ids_sorted_decreasing[i]][0]))
            
            unaligned_art_labels[i, :len(batch[ids_sorted_decreasing[i]][10]), :] = torch.from_numpy(batch[ids_sorted_decreasing[i]][10])
            
            phnid_padded[i, :len(batch[ids_sorted_decreasing[i]][4])] = torch.from_numpy(batch[ids_sorted_decreasing[i]][4])
            phone_lens.append(batch[ids_sorted_decreasing[i]][11])

            
        
        return ema_padded, t(ema_lens), mfcc_padded, t(mfcc_lens), art_labels_padded, phnid_padded, phnid_align_padded, art_weights_padded, weights_mask_padded, unaligned_art_labels, t(phone_lens), feats_padded, names
        # return ema_padded, t(ema_lens), mfcc_padded, t(mfcc_lens), art_labels_padded, phnid_padded, phnid_align_padded, art_weights_padded, weights_mask_padded, t(phone_lens), names                                                                                
                                                                                         
                                                                                           
        
def t(arr):
    if isinstance(arr, list):
        return torch.from_numpy(np.array(arr))
    elif isinstance(arr, np.ndarray):
        return torch.from_numpy(arr)
    else:
        raise NotImplementedError        
        
        
        
        