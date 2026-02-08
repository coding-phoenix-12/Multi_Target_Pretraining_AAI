from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import *
from common.logger import logger
import numpy as np

class FastSpeech(nn.Module):
    def __init__(self, n_mel_channels, padding_idx,input_dim,
                 symbols_embedding_dim, in_fft_n_layers, in_fft_n_heads,
                 in_fft_d_head,
                 in_fft_conv1d_kernel_size, in_fft_conv1d_filter_size,
                 in_fft_output_size, p_in_fft_dropout, p_in_fft_dropatt, 
                 p_in_fft_dropemb,out_fft_output_size,
                 out_fft_n_layers, out_fft_n_heads, out_fft_d_head,
                 out_fft_conv1d_kernel_size, out_fft_conv1d_filter_size,
                 p_out_fft_dropout, p_out_fft_dropatt, p_out_fft_dropemb,
                 n_speakers, speaker_emb_weight, tag, conf, predict=None):
        super(FastSpeech, self).__init__()
        
        self.conf = conf
        self.tag = tag
        self.predict = predict
        self.op_params = []
        
        
        if self.conf.common.inputs == "mfcc":
            self.mfcc_proj = nn.Linear(13, symbols_embedding_dim)
        elif self.conf.common.inputs == "feats":
            self.mfcc_proj = nn.Linear(768, symbols_embedding_dim)

        self.layer1 = FFTransformer(
            n_layer=4, n_head=in_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=in_fft_d_head,
            d_inner=in_fft_conv1d_filter_size,
            kernel_size=in_fft_conv1d_kernel_size,
            dropout=p_in_fft_dropout,
            dropatt=p_in_fft_dropatt,
            dropemb=p_in_fft_dropemb,
            embed_input=False,
            d_embed=symbols_embedding_dim,
            padding_idx=padding_idx)
        

        
        self.layer2 = FFTransformer(
            n_layer=4, n_head=out_fft_n_heads,
            d_model=symbols_embedding_dim,
            d_head=out_fft_d_head,
            d_inner=out_fft_conv1d_filter_size,
            kernel_size=out_fft_conv1d_kernel_size,
            dropout=p_out_fft_dropout,
            dropatt=p_out_fft_dropatt,
            dropemb=p_out_fft_dropemb,
            embed_input=False,
            d_embed=symbols_embedding_dim
        )
        
        self.emaj_proj = nn.Linear(symbols_embedding_dim, 12)
        
        self.op_params += [*self.layer1.parameters(), *self.layer2.parameters(), 
                          *self.mfcc_proj.parameters(), *self.emaj_proj.parameters()
                          ]
        
    def forward(self, inputs, cond=None, infer=False):
        (mfcc_padded, ema_lens, ema_padded, mfcc_lens, feats_padded) = inputs
        
        inputs = 0
        phones_pred = 0
        
        if self.conf.common.inputs == "mfcc":
            enc_in = self.mfcc_proj(mfcc_padded)
        elif self.conf.common.inputs == "feats":
            enc_in = self.mfcc_proj(feats_padded)

        enc_out1, mask = self.layer1(enc_in, mfcc_lens)
        enc_out2, mask = self.layer2(enc_out1, mfcc_lens)
        
        feat_out = self.emaj_proj(enc_out2)
        
        feat_out_dash = [feat_out]
        
        return feat_out, ema_lens, mask, feat_out_dash
    
    
    def loss(self, targets, outputs, infer=False):
        loss_dict = {}
        ema_padded, ema_lens, mfcc_padded, mfcc_lens, art_labels_padded, phnid_padded, phnid_align_padded, art_weights_padded, weights_mask_padded, unaligned_art_labels, phone_lens = targets
        feat_out, ema_lens, mask, feat_out_dash = outputs
       
        mask = mask.float()
        
        ema_loss = F.mse_loss(feat_out, ema_padded, reduction='none')
        ema_loss = ema_loss * mask
        ema_loss = ema_loss.sum(1) / mask.sum(1)
        ema_loss = ema_loss.mean(-1)
        ema_loss = ema_loss.mean()
        
        loss_dict['ema_loss'] = ema_loss
        loss_dict["total_loss"] = ema_loss
        loss_dict[self.tag] = ema_loss
        
        return loss_dict
    
    
    def optimize(self, loss_dict, optimizer):
        torch.nn.utils.clip_grad_norm_(self.op_params, 5)
        optimizer[0].zero_grad()
        loss_dict["total_loss"].backward()
        optimizer[0].step()