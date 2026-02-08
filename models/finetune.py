from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import *
from common.logger import logger
import numpy as np
from models.pretrain import FastSpeech

class FastSpeechft(nn.Module):
    def __init__(self, n_mel_channels, padding_idx,input_dim,
                 symbols_embedding_dim, in_fft_n_layers, in_fft_n_heads,
                 in_fft_d_head,
                 in_fft_conv1d_kernel_size, in_fft_conv1d_filter_size,
                 in_fft_output_size, p_in_fft_dropout, p_in_fft_dropatt, 
                 p_in_fft_dropemb,out_fft_output_size,
                 out_fft_n_layers, out_fft_n_heads, out_fft_d_head,
                 out_fft_conv1d_kernel_size, out_fft_conv1d_filter_size,
                 p_out_fft_dropout, p_out_fft_dropatt, p_out_fft_dropemb,
                 n_speakers, speaker_emb_weight, tag, conf, model_conf):
        super(FastSpeechft, self).__init__()
        
        self.conf = conf
        self.tag = tag
        self.op_params = []
        
        
        self.pre_model = FastSpeech(n_mel_channels=12,
                                        tag=tag,
                                        input_dim=input_dim,
                                        padding_idx=padding_idx,
                                        n_speakers=n_speakers,
                                        conf=conf,
                                        **model_conf
                                        ).to(self.conf.common.device)
        
        self.pre_model.load_state_dict(torch.load("saved_models/" + conf.common.load_models[0], map_location="cpu")["model_state_dict"], strict=False)
        self.ema_proj = nn.Linear(out_fft_output_size, 12, bias=True)
        
        
        self.op_params = [*self.ema_proj.parameters(), *self.pre_model.parameters()]
        
        
        
    def forward(self, inputs, cond=None, infer=False):
        (mfcc_padded, ema_lens, ema_padded, mfcc_lens, feats_padded) = inputs
        
        inputs = 0
        phones_pred = 0
        
        if self.conf.common.inputs == "mfcc":
            enc_in = self.pre_model.mfcc_proj(mfcc_padded)
        elif self.conf.common.inputs == "feats":
            enc_in = self.pre_model.mfcc_proj(feats_padded)
        
        # enc_in = self.pre_model.mfcc_proj(mfcc_padded)
        enc_out1, mask = self.pre_model.layer1(enc_in, mfcc_lens)
        enc_out2, mask = self.pre_model.layer2(enc_out1, mfcc_lens)
        
            
        
        feat_out = self.ema_proj(enc_out2)
        
        feat_out_dash = [feat_out]
        
        return feat_out, ema_lens, mask, feat_out_dash
    
    
    def loss(self, targets, outputs, infer=False):
        loss_dict = {}
        ema_padded, ema_lens, mfcc_padded, mfcc_lens, art_labels_padded, phnid_padded, phnid_align_padded, art_weights_padded, weights_mask_padded, unaligned_art_labels, phone_lens = targets
        feat_out, ema_lens, mask, feat_out_dash = outputs
        
       
        mask = mask.float().squeeze(-1)
 
    
        mask = mask.unsqueeze(-1)
        ema_loss = F.mse_loss(feat_out, ema_padded, reduction='none')
        ema_loss = ema_loss * mask
        ema_loss = ema_loss.sum(1) / mask.sum(1)
        ema_loss = ema_loss.mean()
        loss_dict['ema_loss'] = ema_loss
        
        loss_dict['total_loss'] = ema_loss
        loss_dict[self.tag] = ema_loss

        loss_dict["track"] = ema_loss
        
        
        return loss_dict
        
        
        
        
        
        
    def optimize(self, loss_dict, optimizer):
        torch.nn.utils.clip_grad_norm_(self.op_params, 5)
        optimizer[0].zero_grad()
        loss_dict["total_loss"].backward()
        optimizer[0].step()
        
        
        
    def freeze_layers(self, model):
        for param in model.parameters():
            param.requires_grad = False
         