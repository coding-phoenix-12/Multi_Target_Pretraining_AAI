from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import *
from common.logger import logger
import numpy as np
from models.fastspeech import FastSpeech

class HierFS(nn.Module):
    def __init__(self, n_mel_channels, padding_idx,input_dim,
                 symbols_embedding_dim, in_fft_n_layers, in_fft_n_heads,
                 in_fft_d_head, pyannote_sub_embed, sub_embed_loc,
                 in_fft_conv1d_kernel_size, in_fft_conv1d_filter_size,
                 in_fft_output_size, p_in_fft_dropout, p_in_fft_dropatt, 
                 p_in_fft_dropemb,out_fft_output_size,
                 out_fft_n_layers, out_fft_n_heads, out_fft_d_head,
                 out_fft_conv1d_kernel_size, out_fft_conv1d_filter_size,
                 p_out_fft_dropout, p_out_fft_dropatt, p_out_fft_dropemb,
                 n_speakers, speaker_emb_weight, use_one_hot_spk_embed, tag, conf, model_conf):
        super(HierFS, self).__init__()
        
        self.conf = conf
        self.tag = tag
        self.op_params = []
        
        
        self.models = {}
        for model_name in self.conf.common.load_models:
            model = FastSpeech(n_mel_channels=12,
                                            tag=tag,
                                            input_dim=input_dim,
                                            use_one_hot_spk_embed=use_one_hot_spk_embed,
                                            pyannote_sub_embed=pyannote_sub_embed,
                                            sub_embed_loc=sub_embed_loc,
                                            padding_idx=padding_idx,
                                            n_speakers=n_speakers,
                                            conf=conf,
                                            predict=model_name,
                                            **model_conf
                                            ).to(self.conf.common.device)
            model.load_state_dict(torch.load("saved_models/" + model_name)["model_state_dict"])
            self.models[model_name] = model
            self.freeze_layers(self.models[model_name])
            
            
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
        
        self.layer1_weight = nn.Linear(len(self.models.keys()) + 1, 1)

        
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
        
        self.layer2_weight = nn.Linear(len(self.models.keys()) + 1, 1)
        
        
        self.ema_proj = nn.Linear(symbols_embedding_dim, 12, bias=True)
        
        self.op_params += [
                            *self.layer1.parameters(), *self.layer2.parameters(), 
                            *self.mfcc_proj.parameters(), *self.ema_proj.parameters(),
                            *self.layer1_weight.parameters(), *self.layer2_weight.parameters()
                          ]
        
        
    def forward(self, inputs, cond=None, infer=False):
        (mfcc_padded, ema_lens, ema_padded, mfcc_lens) = inputs
        
        inputs = 0
        phones_pred = 0

        layer1_out_list = []
        
        for model in self.models.keys():
            model_mfcc_embed = self.models[model].mfcc_proj(mfcc_padded)
            layer1_out, mask = self.models[model].layer1(model_mfcc_embed, mfcc_lens)
            layer1_out_list.append(layer1_out)
            
        mfcc_embed = self.mfcc_proj(mfcc_padded)
        layer1_out, mask = self.layer1(mfcc_embed, mfcc_lens)
        layer1_out_list.append(layer1_out)
        
        layer1_out_stack = torch.stack(layer1_out_list, dim=-1)
        layer1_out = self.layer1_weight(layer1_out_stack).squeeze(-1)
        
        layer2_out_list = []
        for model_idx, model in enumerate(self.models.keys()):
            layer2_out, mask = self.models[model].layer2(layer1_out_list[model_idx], mfcc_lens)
            layer2_out_list.append(layer2_out)
            
        layer2_out, mask = self.layer2(mfcc_embed, mfcc_lens)
        layer2_out_list.append(layer2_out)
        
        layer2_out_stack = torch.stack(layer2_out_list, dim=-1)
        layer2_out = self.layer2_weight(layer2_out_stack).squeeze(-1)
        
        feat_out = self.ema_proj(layer2_out)
        
        feat_out_dash = []
        
        for model_idx, model in enumerate(self.models.keys()):
            feat_out_dash.append(self.models[model].phone_pred_layer(layer2_out_list[model_idx]))
            
        return feat_out, ema_lens, mask, feat_out_dash
        

        
    def loss(self, targets, outputs):
        loss_dict = {}
        ema_padded, ema_lens, mfcc_padded, mfcc_lens, art_labels_padded = targets
        feat_out, ema_lens, mask, feat_out_dash = outputs
       
        mask = mask.float()
        
        ema_loss = F.mse_loss(feat_out, ema_padded, reduction='none')
        ema_loss = ema_loss * mask
        ema_loss = ema_loss.sum(1) / mask.sum(1)
        ema_loss = ema_loss.mean()
        
        mask = mask.squeeze(-1)
        
        loss_dict['ema_loss'] = ema_loss
        
        for model_idx, model in enumerate(self.models.keys()):
            label_loss = F.cross_entropy(feat_out_dash[model_idx].permute(0, 2, 1), art_labels_padded[:,:, model_idx].long(), reduction='none')
            label_loss = label_loss 
            label_loss = label_loss.sum(1) / mask.sum(1)
            loss_dict[model + "_loss"] = label_loss.mean()
            
            acc = (torch.argmax(feat_out_dash[model_idx], dim=-1) == art_labels_padded[:,:,model_idx].long()).float()
            acc = acc * mask
            acc = acc.sum(1) / mask.sum(1)
            acc = acc.mean()
            loss_dict[model + "_accuracy"] = acc
            
        loss_dict['total_loss'] = ema_loss
        loss_dict[self.tag] = ema_loss
        
        return loss_dict
        
    
    def optimize(self, loss_dict, optimizer):
        torch.nn.utils.clip_grad_norm_(self.op_params, 5)
        optimizer[0].zero_grad()
        loss_dict["total_loss"].backward()
        optimizer[0].step()
        
        
        
    def freeze_layers(self, model):
        for param in model.parameters():
            param.requires_grad = False
         