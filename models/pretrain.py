from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import *
from common.logger import logger
import numpy as np
from torchaudio.models.decoder import ctc_decoder
import jiwer

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
        
        if "place" in self.conf.common.predict:
            self.place_proj = nn.Linear(symbols_embedding_dim, 12)
            self.op_params += [*self.place_proj.parameters()]
            if self.conf.common.loss == "ctc":
                self.place_ctc_decoder = ctc_decoder(lexicon=None, tokens=[str(x) for x in range(11)], blank_token='11', sil_token="9")

        if "manner" in self.conf.common.predict:
            self.manner_proj = nn.Linear(symbols_embedding_dim, 10)
            self.op_params += [*self.manner_proj.parameters()] 
            if self.conf.common.loss == "ctc":
                self.manner_ctc_decoder = ctc_decoder(lexicon=None, tokens=[str(x) for x in range(10)], blank_token='9', sil_token="8")

        if "height" in self.conf.common.predict:
            self.height_proj = nn.Linear(symbols_embedding_dim, 11)
            self.op_params += [*self.height_proj.parameters()]
            if self.conf.common.loss == "ctc":
                self.height_ctc_decoder = ctc_decoder(lexicon=None, tokens=[str(x) for x in range(11)], blank_token='10', sil_token="9")

        if "backness" in self.conf.common.predict:
            self.backness_proj = nn.Linear(symbols_embedding_dim, 7)
            self.op_params += [*self.backness_proj.parameters()]
            if self.conf.common.loss == "ctc":
                self.backness_ctc_decoder = ctc_decoder(lexicon=None, tokens=[str(x) for x in range(7)], blank_token='6', sil_token="5")

            
        if "phones" in self.conf.common.predict:
            self.phone_pred_layer = nn.Linear(symbols_embedding_dim, 45)
            self.op_params += [*self.phone_pred_layer.parameters()]
            if self.conf.common.loss == "ctc":
                self.phones_ctc_decoder = ctc_decoder(lexicon=None, tokens=[str(x) for x in range(45)], blank_token='44', sil_token="16")
            
        if "weights" in self.conf.common.predict:
            self.weight_pred_layer = nn.Linear(symbols_embedding_dim, 12)
            self.op_params += [*self.weight_pred_layer.parameters()]


        self.op_params += [
                        *self.layer1.parameters(), *self.layer2.parameters(), 
                        *self.mfcc_proj.parameters()
                    ]


    def forward(self, inputs, cond=None, infer=False):
        (mfcc_padded, ema_lens, ema_padded, mfcc_lens, feats_padded) = inputs
        
        inputs = 0
        phones_pred = 0

      
        if self.conf.common.inputs == "mfcc":
            enc_in = self.mfcc_proj(mfcc_padded)
        elif self.conf.common.inputs == "feats":
            enc_in = self.mfcc_proj(feats_padded)

        layer1_out, mask = self.layer1(enc_in, mfcc_lens)
        
        layer2_out, mask = self.layer2(layer1_out, mfcc_lens)


        feat_out_dash = {}
        if "place" in self.conf.common.predict:
            place_pred = self.place_proj(layer2_out)
            feat_out_dash["place"] = place_pred
        
        if "manner" in self.conf.common.predict:
            manner_pred = self.manner_proj(layer2_out)
            feat_out_dash["manner"] = manner_pred

        if "height" in self.conf.common.predict:
            height_pred = self.height_proj(layer2_out)
            feat_out_dash["height"] = height_pred

        if "backness" in self.conf.common.predict:
            backness_pred = self.backness_proj(layer2_out)
            feat_out_dash["backness"] = backness_pred

            
        if "phones" in self.conf.common.predict:
            phones_pred = self.phone_pred_layer(layer2_out)
            feat_out_dash["phones"] = phones_pred
            
        if "weights" in self.conf.common.predict:
            weight_pred = self.weight_pred_layer(layer2_out)
            feat_out_dash["weights"] = weight_pred

        feat_out = [feat_out_dash[key] for key in feat_out_dash]
        
        
        return feat_out, ema_lens, mask, feat_out_dash
    
    
    
    def loss(self, targets, outputs, infer=False):
        loss_dict = {}
        ema_padded, ema_lens, mfcc_padded, mfcc_lens, art_labels_padded, phnid_padded, phnid_align_padded, art_weights_padded, weights_mask_padded, unaligned_art_labels, phone_lens = targets
        feat_out, ema_lens, mask, feat_out_dash = outputs
       
        mask = mask.float().squeeze(-1)
        phones_mask = mask_from_lens(phone_lens).float()
       
        art_labels_padded = art_labels_padded.long()
        total_loss = 0
        loss_dict["track"] = 0

        if self.conf.common.loss == "ce":
            if "place" in self.conf.common.predict:
                
                place_loss = F.cross_entropy(feat_out_dash["place"].permute(0, 2, 1), art_labels_padded[:,:, 0], reduction='none')
                place_loss = (place_loss * mask).sum(1) / mask.sum(1)
                place_loss = place_loss.mean()
                loss_dict["place_loss"] = place_loss
                total_loss += place_loss

                place_acc = (feat_out_dash["place"].argmax(2) == art_labels_padded[:, :, 0]).float()
                place_acc = (place_acc * mask).sum(1) / mask.sum(1)
                place_acc = place_acc.mean()
                loss_dict["place_accuracy"] = place_acc
                
                loss_dict["track"] += -place_acc
                # hyp = self.place_ctc_decoder.decode(feat_out_dash["place"], mfcc_lens)
                
                

            if "manner" in self.conf.common.predict:

                manner_loss = F.cross_entropy(feat_out_dash["manner"].permute(0, 2, 1), art_labels_padded[:, :, 1], reduction='none')
                manner_loss = (manner_loss * mask).sum(1) / mask.sum(1)
                manner_loss = manner_loss.mean()
                loss_dict["manner_loss"] = manner_loss
                total_loss += manner_loss

                manner_acc = (feat_out_dash["manner"].argmax(2) == art_labels_padded[:, :, 1]).float()
                manner_acc = (manner_acc * mask).sum(1) / mask.sum(1)
                manner_acc = manner_acc.mean()
                loss_dict["manner_accuracy"] = manner_acc
                
                loss_dict["track"] += -manner_acc

            if "height" in self.conf.common.predict:
                height_loss = F.cross_entropy(feat_out_dash["height"].permute(0, 2, 1), art_labels_padded[:, :, 2], reduction='none')
                height_loss = (height_loss * mask).sum(1) / mask.sum(1)
                height_loss = height_loss.mean()
                loss_dict["height_loss"] = height_loss
                total_loss += height_loss

                height_acc = (feat_out_dash["height"].argmax(2) == art_labels_padded[:, :, 2]).float()
                height_acc = (height_acc * mask).sum(1) / mask.sum(1)
                height_acc = height_acc.mean()
                loss_dict["height_accuracy"] = height_acc
                
                loss_dict["track"] += -height_acc

            if "backness" in self.conf.common.predict:
                backness_loss = F.cross_entropy(feat_out_dash["backness"].permute(0, 2, 1), art_labels_padded[:, :, 3], reduction='none')
                backness_loss = (backness_loss * mask).sum(1) / mask.sum(1)
                backness_loss = backness_loss.mean()
                loss_dict["backness_loss"] = backness_loss
                total_loss += backness_loss

                backness_acc = (feat_out_dash["backness"].argmax(2) == art_labels_padded[:, :, 3]).float()
                backness_acc = (backness_acc * mask).sum(1) / mask.sum(1)
                backness_acc = backness_acc.mean()
                loss_dict["backness_accuracy"] = backness_acc
                
                loss_dict["track"] += -backness_acc
                
                
            if "phones" in self.conf.common.predict:
                phone_loss  = F.cross_entropy(feat_out_dash["phones"].permute(0, 2, 1), phnid_align_padded.long(), reduction='none')
                phone_loss = (phone_loss * mask).sum(1) / mask.sum(1)
                phone_loss = phone_loss.mean(0)
                loss_dict['phone_loss'] = phone_loss
                total_loss += phone_loss
                
                phone_acc = (torch.argmax(feat_out_dash["phones"], dim=-1) == phnid_align_padded.long()).float()
                phone_acc = (phone_acc * mask).sum(1) / mask.sum(1)
                phone_acc = phone_acc.mean(0)
                loss_dict['phone_accuracy'] = phone_acc
                
                loss_dict["track"] += -phone_acc
                
                total_loss += phone_loss
                
                
        elif self.conf.common.loss == "ctc":
            if "place" in self.conf.common.predict:    
                # place_loss = F.cross_entropy(feat_out_dash["place"].permute(0, 2, 1), art_labels_padded[:,:, 0], reduction='none')
                place_loss = F.ctc_loss(F.log_softmax(feat_out_dash["place"].permute(1, 0, 2), dim=-1), unaligned_art_labels[:, :, 0], mfcc_lens.int(), phone_lens.int(), blank=10, reduction='mean')
                # place_loss = (place_loss * mask).sum(1) / mask.sum(1)
                # place_loss = place_loss.mean()
                loss_dict["place_loss"] = place_loss
                total_loss += place_loss

                # place_acc = (feat_out_dash["place"].argmax(2) == art_labels_padded[:, :, 0]).float()
                hyp = self.place_ctc_decoder.decode(feat_out_dash["place"].detach().cpu(), mfcc_lens)
                out = [x[0][0] for x in hyp]
                gt = []
                pred = []
                
                for out_idx, out_b in enumerate(out):
                    gt.append("".join([str(x) for x in out_b]))
                    pred.append("".join([str(x) for x in unaligned_art_labels[out_idx, :mfcc_lens[out_idx], 0].cpu().numpy()]))
                    
                place_acc = jiwer.cer(gt, pred)
                    
                loss_dict["place_accuracy"] = place_acc
                
                loss_dict["track"] += place_loss

            if "manner" in self.conf.common.predict:

                # manner_loss = F.cross_entropy(feat_out_dash["manner"].permute(0, 2, 1), art_labels_padded[:, :, 1], reduction='none')
                # place_loss = F.ctc_loss(F.log_softmax(feat_out_dash["manner"].permute(1, 0, 2), dim=-1), art_labels_padded[:, :, 1], mfcc_lens, ema_lens, blank=10, reduction='none')
                manner_loss = F.ctc_loss(F.log_softmax(feat_out_dash["mannner"].permute(1, 0, 2), dim=-1), unaligned_art_labels[:, :, 1], mfcc_lens.int(), phone_lens.int(), blank=10, reduction='mean')
                # manner_loss = (manner_loss * mask).sum(1) / mask.sum(1)
                # manner_loss = manner_loss.mean()
                loss_dict["manner_loss"] = manner_loss
                total_loss += manner_loss

                hyp = self.manner_ctc_decoder.decode(feat_out_dash["manner"].detach().cpu(), mfcc_lens)
                out = [x[0][0] for x in hyp]
                gt = []
                pred = []
                
                for out_idx, out_b in enumerate(out):
                    gt.append("".join([str(x) for x in out_b]))
                    pred.append("".join([str(x) for x in unaligned_art_labels[out_idx, :mfcc_lens[out_idx], 0].cpu().numpy()]))
                    
                manner_acc = jiwer.cer(gt, pred)
                loss_dict["manner_accuracy"] = manner_acc
                
                loss_dict["track"] += manner_loss

            if "height" in self.conf.common.predict:
                # height_loss = F.cross_entropy(feat_out_dash["height"].permute(0, 2, 1), art_labels_padded[:, :, 2], reduction='none')
                height_loss = F.ctc_loss(F.log_softmax(feat_out_dash["height"].permute(1, 0, 2), dim=-1), unaligned_art_labels[:, :, 2], mfcc_lens.int(), phone_lens.int(), blank=10, reduction='mean')
                # height_loss = (height_loss * mask).sum(1) / mask.sum(1)
                # height_loss = height_loss.mean()
                loss_dict["height_loss"] = height_loss
                total_loss += height_loss

                hyp = self.height_ctc_decoder.decode(feat_out_dash["height"].detach().cpu(), mfcc_lens)
                out = [x[0][0] for x in hyp]
                gt = []
                pred = []
                
                for out_idx, out_b in enumerate(out):
                    gt.append("".join([str(x) for x in out_b]))
                    pred.append("".join([str(x) for x in unaligned_art_labels[out_idx, :mfcc_lens[out_idx], 0].cpu().numpy()]))
                    
                height_acc = jiwer.cer(gt, pred)
                loss_dict["height_accuracy"] = height_acc
                
                loss_dict["track"] += height_loss

            if "backness" in self.conf.common.predict:
                # backness_loss = F.cross_entropy(feat_out_dash["backness"].permute(0, 2, 1), art_labels_padded[:, :, 3], reduction='none')
                backness_loss = F.ctc_loss(F.log_softmax(feat_out_dash["backness"].permute(1, 0, 2), dim=-1), unaligned_art_labels[:, :, 3], mfcc_lens.int(), phone_lens.int(), blank=10, reduction='mean')
                # backness_loss = (backness_loss * mask).sum(1) / mask.sum(1)
                # backness_loss = backness_loss.mean()
                loss_dict["backness_loss"] = backness_loss
                total_loss += backness_loss

                hyp = self.backness_ctc_decoder.decode(feat_out_dash["backness"].detach().cpu(), mfcc_lens)
                out = [x[0][0] for x in hyp]
                gt = []
                pred = []
                
                for out_idx, out_b in enumerate(out):
                    gt.append("".join([str(x) for x in out_b]))
                    pred.append("".join([str(x) for x in unaligned_art_labels[out_idx, :mfcc_lens[out_idx], 0].cpu().numpy()]))
                    
                backness_acc = jiwer.cer(gt, pred)
                loss_dict["backness_accuracy"] = backness_acc
                
                loss_dict["track"] += backness_loss
                
                
            if "phones" in self.conf.common.predict:
                pred = feat_out_dash["phones"].log_softmax(-1)
                pred = pred * mask.unsqueeze(-1)
    
                
                phone_loss = F.ctc_loss(pred.permute(1, 0, 2), phnid_padded, mfcc_lens.int(), phone_lens.int(), blank=44, reduction='sum', zero_infinity=True)
                # phone_loss = (phone_loss * mask).sum(1) / mask.sum(1)
                # phone_loss = phone_loss.mean(0)
                loss_dict['phone_loss'] = phone_loss
                total_loss += phone_loss
                
                if infer:
                    hyp = self.phones_ctc_decoder(pred.detach().cpu(), mfcc_lens.detach().cpu())
                    out = [x[0][0] for x in hyp]
                    gt = []
                    pred_list = []
                    
                
                    for out_idx, out_b in enumerate(out):
                        pred_list.append("".join([str(x) for x in out_b][1:-1]))
                        gt.append("".join([str(x) for x in phnid_padded[out_idx, :mfcc_lens[out_idx].int()].cpu().numpy()]))
                    
                    phone_acc = jiwer.cer(gt, pred_list)
                    loss_dict['phone_accuracy'] = torch.tensor(phone_acc)

                else:
                    loss_dict['phone_accuracy'] = phone_loss
                
                total_loss += phone_loss
                
                loss_dict["track"] += phone_loss
                

        if "weights" in self.conf.common.predict:
            weight_loss = F.binary_cross_entropy_with_logits(feat_out_dash["weights"], art_weights_padded, reduction='none')
            weights_sum = weights_mask_padded.sum(1)
            weights_sum[weights_sum == 0] = 1
            weight_loss = (weight_loss * weights_mask_padded).sum(1) / weights_sum
            weight_loss = weight_loss.mean()
            loss_dict['weight_loss'] = weight_loss
            
            weight_acc = (torch.round(torch.sigmoid(feat_out_dash["weights"])) == art_weights_padded).float()
            weight_acc = (weight_acc * weights_mask_padded).sum(1) / weights_sum
            weight_acc = weight_acc.mean(-1)
            weight_acc = weight_acc.mean(0)
            loss_dict['weight_accuracy'] = weight_acc
            
            loss_dict["track"] += weight_acc
            
            
            total_loss += weight_loss
            
        

        loss_dict[self.tag] = total_loss
        loss_dict["total_loss"] = loss_dict[self.tag]

        return loss_dict
    
    def optimize(self, loss_dict, optimizer):
        torch.nn.utils.clip_grad_norm_(self.op_params, 5)
        optimizer[0].zero_grad()
        loss_dict["total_loss"].backward()
        optimizer[0].step()


