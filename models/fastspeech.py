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
        
        
        self.mfcc_proj = nn.Linear(13, symbols_embedding_dim)
        
        self.layer1 = FFTransformer(
            n_layer=1, n_head=in_fft_n_heads,
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
            n_layer=1, n_head=out_fft_n_heads,
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

        
            
        # self.ema_proj = nn.Linear(symbols_embedding_dim, 12, bias=True)
        if self.conf.common.predict == 'place' or self.predict=='place':
            self.phone_pred_layer = nn.Linear(symbols_embedding_dim, 11)
            self.op_params += [*self.phone_pred_layer.parameters()]
        
        if self.conf.common.predict == 'manner' or self.predict=='manner':
            self.phone_pred_layer = nn.Linear(symbols_embedding_dim, 9)
            self.op_params += [*self.phone_pred_layer.parameters()]
            
        if self.conf.common.predict == 'height' or self.predict=='height':
            self.phone_pred_layer = nn.Linear(symbols_embedding_dim, 10)
            self.op_params += [*self.phone_pred_layer.parameters()]
        
        if self.conf.common.predict == 'backness' or self.predict=='backness':
            self.phone_pred_layer = nn.Linear(symbols_embedding_dim, 6)
            self.op_params += [*self.phone_pred_layer.parameters()]
        
        
        self.op_params += [*self.layer1.parameters(), *self.layer2.parameters(), 
                          *self.mfcc_proj.parameters(), 
                          ]
        

    def forward(self, inputs, cond=None, infer=False):
        (mfcc_padded, ema_lens, ema_padded, mfcc_lens) = inputs
        
        inputs = 0
        phones_pred = 0

      
        mfcc_embed = self.mfcc_proj(mfcc_padded)

       
        layer1_out, mask = self.layer1(mfcc_embed, mfcc_lens)
        
        layer2_out, mask = self.layer2(layer1_out, mfcc_lens)

        if self.conf.common.predict == 'place' or self.predict=='place':
            feat_out = self.phone_pred_layer(layer2_out)
            
        if self.conf.common.predict == 'manner' or self.predict=='manner':
            feat_out = self.phone_pred_layer(layer2_out)
            
        if self.conf.common.predict == 'height' or self.predict=='height':
            feat_out = self.phone_pred_layer(layer2_out)
            
        if self.conf.common.predict == 'backness' or self.predict=='backness':
            feat_out = self.phone_pred_layer(layer2_out)
            
        

        
        
        feat_out_dash = [layer1_out, layer2_out]
        return feat_out, ema_lens, mask, feat_out_dash
    
    
    def loss(self, targets, outputs):
        loss_dict = {}
        ema_padded, ema_lens, mfcc_padded, mfcc_lens, art_labels_padded = targets
        feat_out, ema_lens, mask, feat_out_dash = outputs
       
        mask = mask.float()
        
        if self.conf.common.predict == 'place' or self.predict=='place':
            place_loss = F.cross_entropy(feat_out.permute(0, 2, 1), art_labels_padded[:,:, 0].long(), reduction='none', ignore_index=0)
            place_loss = place_loss * mask.squeeze(-1)
            place_loss = place_loss.sum(1) / mask.sum(1).squeeze(-1)
            loss_dict["place_loss"] = place_loss.mean()
            
            place_accuracy = (torch.argmax(feat_out, dim=2).detach() == art_labels_padded[:,:,0]).float()
            place_accuracy = place_accuracy * mask.squeeze(-1)
            place_accuracy = place_accuracy.sum(1) / mask.sum(1).squeeze(-1)
            loss_dict["place_accuracy"] = place_accuracy.mean()
            
            loss_dict[self.tag] = loss_dict["place_loss"]
            loss_dict["total_loss"] = loss_dict["place_loss"]
            
        if self.conf.common.predict == 'manner' or self.predict=='manner':
            manner_loss = F.cross_entropy(feat_out.permute(0, 2, 1), art_labels_padded[:,:, 1].long(), reduction='none', ignore_index=0)
            manner_loss = manner_loss * mask.squeeze(-1)
            manner_loss = manner_loss.sum(1) / mask.sum(1).squeeze(-1)
            loss_dict["manner_loss"] = manner_loss.mean()
            
            manner_accuracy = (torch.argmax(feat_out, dim=2).detach() == art_labels_padded[:,:,1]).float()
            manner_accuracy = manner_accuracy * mask.squeeze(-1)
            manner_accuracy = manner_accuracy.sum(1) / mask.sum(1).squeeze(-1)
            loss_dict["manner_accuracy"] = manner_accuracy.mean()
            
            loss_dict[self.tag] = loss_dict["manner_loss"]
            loss_dict["total_loss"] = loss_dict["manner_loss"]
            
        if self.conf.common.predict == 'height'or self.predict=='height':
            height_loss = F.cross_entropy(feat_out.permute(0, 2, 1), art_labels_padded[:,:, 2].long(), reduction='none', ignore_index=0)
            height_loss = height_loss * mask.squeeze(-1)
            height_loss = height_loss.sum(1) / mask.sum(1).squeeze(-1)
            loss_dict["height_loss"] = height_loss.mean()
            
            height_accuracy = (torch.argmax(feat_out, dim=2).detach() == art_labels_padded[:,:,2]).float()
            height_accuracy = height_accuracy * mask.squeeze(-1)
            height_accuracy = height_accuracy.sum(1) / mask.sum(1).squeeze(-1)
            loss_dict["height_accuracy"] = height_accuracy.mean()
            
            loss_dict[self.tag] = loss_dict["height_loss"]
            loss_dict["total_loss"] = loss_dict["height_loss"]
            
        if self.conf.common.predict == 'backness' or self.predict=='backness':
            backness_loss = F.cross_entropy(feat_out.permute(0, 2, 1), art_labels_padded[:,:, 3].long(), reduction='none', ignore_index=0)
            backness_loss = backness_loss * mask.squeeze(-1)
            backness_loss = backness_loss.sum(1) / mask.sum(1).squeeze(-1)
            loss_dict["backness_loss"] = backness_loss.mean()
            
            backness_accuracy = (torch.argmax(feat_out, dim=2).detach() == art_labels_padded[:,:,3]).float()
            backness_accuracy = backness_accuracy * mask.squeeze(-1)
            backness_accuracy = backness_accuracy.sum(1) / mask.sum(1).squeeze(-1)
            loss_dict["backness_accuracy"] = backness_accuracy.mean()
            
            loss_dict[self.tag] = loss_dict["backness_loss"]
            loss_dict["total_loss"] = loss_dict["backness_loss"]
        
        
       
        return loss_dict
    
    def optimize(self, loss_dict, optimizer):
        torch.nn.utils.clip_grad_norm_(self.op_params, 5)
        optimizer[0].zero_grad()
        loss_dict["total_loss"].backward()
        optimizer[0].step()
        
        
    
    