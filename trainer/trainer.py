import os
import torch
import librosa
import numpy as np
import scipy.stats
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
from common.logger import logger
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.common import mask_from_lens
# from common.wandb_logger import WandbLogger
import scipy.interpolate
from scipy.interpolate import interp1d
from torch.optim.lr_scheduler import LambdaLR
import jiwer
import soundfile as sf


import common
from common.optim_wrap import ScheduledOptim


class Operate():
    def __init__(self, params):
        self.patience = int(params.earlystopper.patience)
        self.counter = 0
        self.bestScore = None
        self.earlyStop = False
        self.valMinLoss = np.inf
        self.delta = float(params.earlystopper.delta)
        self.minRun = int(params.earlystopper.minRun)
        self.numEpochs = int(params.common.mode.num_epochs)
        self.modelName = params.common.mode.model
        self.config = params
        # if params.logging.disable:
        #     logger.info('wandb logging disabled')
        #     os.environ['WANDB_MODE'] = 'offline'
        # self.logger = WandbLogger(params)
        self.expmode = params.common.mode.name
        logger.info(f'Predicting {self.expmode}')
        self.best_cc = [-1, 0]
        self.break_mode = params.common.break_mode
        if self.break_mode:
            self.numEpochs = 1
            logger.info('Starting in break mode!')
        self.feats = params.common.mode.feats
        self.train_modes = ['train']
        self.best_er = np.inf
        self.save_check = False
        self.epoch = 0
        
        self.total_cc, self.total_rmse = [], []
        self.place_loss, self.place_acc, self.manner_loss, self.manner_acc, self.height_loss, self.height_acc, self.backness_loss, self.backness_acc = [], [], [], [], [], [], [], []
        self.ema_loss, self.phone_loss, self.phone_acc, self.total_loss = [], [], [], []
        self.weight_loss, self.weight_acc = [], []
        
        self.track_val = []
    
    def esCheck(self):
        score = -self.trackValLoss
        # print(score, self.bestScore)
        if self.epoch>self.minRun:
            if self.bestScore is None:
                self.bestScore = score
                self.saveCheckpoint()
            elif score < self.bestScore + self.delta:
                self.counter += 1
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.earlyStop = True
            else:
                self.bestScore = score
                self.saveCheckpoint()
                self.counter = 0

    def saveCheckpoint(self):
        print("Saving checkpoint", self.config.logging.run_name)
        save_path = 'saved_models'
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer': [self.optimizer[x].state_dict() for x in range(len(self.optimizer))],
            'loss': self.bestScore,
            }, os.path.join(save_path, self.config.logging.run_name))
        
    def loadCheckpoint(self):
        print("Loading checkpoint", self.config.logging.run_name)
        save_path = 'saved_models'
        checkpoint = torch.load(os.path.join(save_path, self.config.logging.run_name), map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.drift_optimizer.load_state_dict(checkpoint['drift_optimizer_state_dict'])
        # self.diffusion_optimizer.load_state_dict(checkpoint['diffusion_optimizer_state_dict'])
        for x in range(len(self.optimizer)):
            self.optimizer[x].load_state_dict(checkpoint['optimizer'][x])
        self.bestScore = checkpoint['loss']
        self.epoch = checkpoint['epoch']
        
    
    def trainloop(self, loader, mode, break_run=False, ):
        if self.break_mode: break_run = True
        if mode == 'train' and not self.config.common.infer: self.model.train()
        # elif mode not in self.train_modes: self.model.eval()
        else: self.model.eval()
        self.reset_iter_metrics()
        losses_to_upload = {'ema':[], 'dur':[], 'total':[]}
        
        with tqdm(loader, unit="batch", mininterval=2) as tepoch:
            for counter, data in enumerate(tepoch):

               
                ema_padded, ema_lens, mfcc_padded, mfcc_lens, art_labels_padded, phnid_padded, phnid_align_padded, art_weights_padded, weights_mask_padded, unaligned_art_labels, phone_lens, feats_padded, names = data
                
                ema_padded, ema_lens, mfcc_padded, mfcc_lens, art_labels_padded, phnid_padded, phnid_align_padded, art_weights_padded, weights_mask_padded, unaligned_art_labels, phone_lens, feats_padded = self.set_device([ema_padded, ema_lens, mfcc_padded, mfcc_lens, art_labels_padded, phnid_padded, phnid_align_padded, art_weights_padded, weights_mask_padded, unaligned_art_labels, phone_lens, feats_padded], ignoreList=[])
            
            
                cond = None
                
            
                inputs = (mfcc_padded, ema_lens, ema_padded, mfcc_lens, feats_padded)
                        
                    
                    
               
                infer = True if mode != 'train' else False
                model_out = self.model(inputs, cond=cond, infer=infer)
                
                feat_out, out_lens, dec_mask, feat_out_dash = model_out
                

                targets = (ema_padded, ema_lens, mfcc_padded, mfcc_lens, art_labels_padded, phnid_padded, phnid_align_padded, art_weights_padded, weights_mask_padded, unaligned_art_labels, phone_lens)
                loss_dict = self.model.loss(targets, model_out, infer=infer)
                
                

                if mode == 'train' and not self.config.common.infer:
                    self.model.optimize(loss_dict, self.optimizer)
                    
                else:
                    self.track_val.append(loss_dict['total_loss'].item())
                    
                    if "place" in self.config.common.predict:
                        self.place_loss.append(loss_dict["place_loss"].item())
                        self.place_acc.append(loss_dict["place_accuracy"].item())
                        
                    if "manner" in self.config.common.predict:
                        self.manner_loss.append(loss_dict["manner_loss"].item())
                        self.manner_acc.append(loss_dict["manner_accuracy"].item())
                        
                    if "height" in self.config.common.predict:
                        self.height_loss.append(loss_dict["height_loss"].item())
                        self.height_acc.append(loss_dict["height_accuracy"].item())
                        
                    if "backness" in self.config.common.predict:
                        self.backness_loss.append(loss_dict["backness_loss"].item())
                        self.backness_acc.append(loss_dict["backness_accuracy"].item())
                        
                        
                    if "phones" in self.config.common.predict:
                        self.phone_loss.append(loss_dict["phone_loss"].item())
                        self.phone_acc.append(loss_dict["phone_accuracy"].item())

                    if "ema" in  self.config.common.predict:
                        self.ema_loss.append(loss_dict["ema_loss"].item())
                        metrics = self.get_cc(ema_padded.detach().cpu(), feat_out.detach().cpu(), ema_lens.detach().cpu().numpy().astype(int).tolist())
                        self.total_cc += metrics[0]
                        self.total_rmse += metrics[1]    
                        
                    if "weights" in self.config.common.predict:
                        self.weight_loss.append(loss_dict["weight_loss"].item())
                        self.weight_acc.append(loss_dict["weight_accuracy"].item())
                        
                        
                        
                        
                        
                        
                self.handle_metrics(loss_dict, mode)
                tepoch.set_postfix(loss=loss_dict["total_loss"].item())
                        
                        
                        
                        
                if break_run:
                    break   

        
        
        # self.dump_cc()
        self.end_of_epoch(ema_padded, feat_out_dash, ema_lens, mode)
        return
                        
                    
                    
    def end_of_epoch(self, ema, out, melLengths, mode):
        if mode not in self.train_modes:
            # self.logger.plot_ema(ema, out, melLengths)
            # self.trackValLoss = sum(self.trackValLoss)/len(self.trackValLoss)
            if "ema" in self.config.common.predict:
                self.trackValLoss = -np.mean(self.total_cc)
            else:
                self.trackValLoss = np.mean(self.track_val)
                
                
        
                
                
        for key in self.epoch_loss_dict:
            self.epoch_loss_dict[key] = sum(self.epoch_loss_dict[key])/len(self.epoch_loss_dict[key])
            

        if "ema" in self.config.common.predict:
            self.cc = np.mean(self.total_cc)
        else:
            self.cc = -np.mean(self.track_val)
        
        
        if mode == 'val' :
            if self.cc > self.best_cc[0]:
                self.best_cc = [self.cc, self.epoch]
                
                
        if mode != 'train':
            if "place" in self.config.common.predict:
                logger.info(f'{mode}, place_loss {round(np.mean(self.place_loss), 4)}, place_acc {round(np.mean(self.place_acc), 4)}')
            if "manner" in self.config.common.predict:
                logger.info(f'{mode}, manner_loss {round(np.mean(self.manner_loss), 4)}, manner_acc {round(np.mean(self.manner_acc), 4)}')
            if "height" in self.config.common.predict:
                logger.info(f'{mode}, height_loss {round(np.mean(self.height_loss), 4)}, height_acc {round(np.mean(self.height_acc), 4)}')
            if "backness" in self.config.common.predict:
                logger.info(f'{mode}, backness_loss {round(np.mean(self.backness_loss), 4)}, backness_acc {round(np.mean(self.backness_acc), 4)}')
            if "phones" in self.config.common.predict:
                logger.info(f'{mode}, phone_loss {round(np.mean(self.phone_loss), 4)}, phone_acc {round(np.mean(self.phone_acc), 4)}')
            
            if "ema" in self.config.common.predict:
                total_cc_std = np.std(self.total_cc)
                total_rmse_std = np.std(self.total_rmse)
                logger.info(f'{mode}, ema_loss {round(np.mean(self.ema_loss), 4)}')
                logger.info(f'{mode}, CC: {round(np.mean(self.total_cc), 4)}({round(total_cc_std, 4)}), rmse {round(np.mean(self.total_rmse), 3)} ({round(total_rmse_std, 3)})')
            
            if "weights" in self.config.common.predict:
                logger.info(f'{mode}, weight_loss {round(np.mean(self.weight_loss), 4)}, weight_acc {round(np.mean(self.weight_acc), 4)}')
           
            
            
    def reset_iter_metrics(self):
        self.epoch_loss_dict = {}
        self.skipped = 0
        self.trackValLoss = []
        self.aai_loss = []
        self.all_mask_lens = 0
        self.total_lens = 0
        self.total_cc, self.total_rmse = [], []
        self.place_loss, self.place_acc, self.manner_loss, self.manner_acc, self.height_loss, self.height_acc, self.backness_loss, self.backness_acc = [], [], [], [], [], [], [], []
        self.ema_loss, self.total_loss = [], []
        
        
        self.phone_loss, self.phone_acc = [], []
        self.weight_loss, self.weight_acc = [], []
        self.track_val = []
        


    def handle_metrics(self, iter_loss_dict, mode):
        for key in iter_loss_dict:
            if f'{key}_{mode}' not in self.epoch_loss_dict:
                self.epoch_loss_dict[f'{key}_{mode}'] = [iter_loss_dict[key].item()]
            else:
                self.epoch_loss_dict[f'{key}_{mode}'].append(iter_loss_dict[key].item())
        # if mode not in self.train_modes:
        #     self.trackValLoss.append(iter_loss_dict["aai"].item())
            
            
    def trainer(self, model, loaders):
        (
        trainLoader, 
        valLoader, 
        testLoader, 
        unseentestloader2, 
        unseentestloader3
        ) = loaders
        

        # if self.config.common.mode.model == "vqvae"  or self.config.common.mode.model == "transformer" or self.config.common.mode.model == "fastspeech_lstm" or self.config.common.mode.model == "whisper":
        fs_optimizer, self.scheduler, self.lr_schedule = self.get_trainers(model.op_params)
        self.optimizer = [fs_optimizer]


            
        # self.lr_schedule = self.get_inverse_sqrt_schedule_with_warmup(self.optimizer, 10, self.config.common.mode.num_epochs)
        self.model = model
        total_params = sum(
	        param.numel() for param in model.parameters()
        )
        logger.info(f'param count :{total_params}')



        if not self.config.common.infer:
            if self.config.common.finetune:
                try:    
                    self.loadCheckpoint()
                except FileNotFoundError:
                    logger.info('No pretrained model found for finetuning')

            # self.saveCheckpoint()
            
            for epoch in range(self.epoch, self.numEpochs):
                print("epoch", epoch)
                print(self.config.logging.run_name)
                self.epoch = epoch
                self.trainloop(trainLoader, 'train')
                
                logger.info("Val")
                self.trainloop(valLoader, 'val')
                

                if self.config.common.mode.early_stop and self.epoch > self.config.common.mode.early_stop_min :
                    self.esCheck()
                    if self.earlyStop:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
                else:
                    self.saveCheckpoint()
    
            
            logger.info('Training completed')

        else:
            self.epoch = 1
            logger.info('Starting Inference')
            try:    
                self.loadCheckpoint()
            except FileNotFoundError:
                logger.info('No pretrained model found for inference')
                
            # self.trainloop(trainLoader, 'train')
            # self.trainloop(valLoader, 'val')
            # logger.summary({'test cc':self.best_cc[0]})
            
            self.trainloop(testLoader, 'test')
            self.unseen_spk_train_cc = self.cc
            self.trainloop(unseentestloader3, 'unseen_spk_test')
            self.unseen_spk_val_cc = self.cc
            
            
    def get_cc(self, ema_, pred_, test_lens):
        ema_ = ema_.permute(0, 2, 1).numpy()
        pred_ = pred_.permute(0, 2, 1).numpy()
        m = []
        rMSE = []
        for j in range(len(pred_)):
            c  = []
            rmse = []
            for k in range(12):
                try:
                    c.append(scipy.stats.pearsonr(ema_[j][k][:test_lens[j]], pred_[j][k][:test_lens[j]])[0])
                    rmse.append(np.sqrt(np.mean(np.square(np.asarray(pred_[j][k][:test_lens[j]])-np.asarray(ema_[j][k][:test_lens[j]])))))
                except:
                    print(pred_)
                    exit()
            m.append(c)
            rMSE.append(rmse)
        return m, rMSE
    
    
    def get_inverse_sqrt_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        """ Create a schedule with a learning rate that linearly increases from some initial learning rate (``--warmup-init-lr``) until the configured
        learning rate (``--lr``). Thereafter we decay proportional to the inverse square root of
        the number of updates.
        """

        def lr_lambda(current_step):
            current_step = max(1, current_step)
            return min(float(current_step)**-0.5, float(current_step)*(float(num_warmup_steps)**-1.5))
        
        return LambdaLR(optimizer, lr_lambda, last_epoch)
    
    def get_trainers(self, model_params):
        if self.config.optimizer.name == 'adam':
            # self.param_groups = [{'params': model.parameters(), 'lr': float(self.config.optimizer.lr)}]
            optimizer = torch.optim.Adam([{'params': model_params, 'lr':float(self.config.optimizer.lr)}], weight_decay=float(self.config.optimizer.weightdecay))
        elif self.config.optimizer.name == 'radam':
            optimizer = torch.optim.RAdam([{'params': model_params, 'lr':float(self.config.optimizer.lr)}], weight_decay=float(self.config.optimizer.weightdecay))

        else:
            raise Exception('Optimizer not found')
        

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 3, factor=0.6)
        if self.config.common.mode.warmup:
            lr_schedule = self.get_inverse_sqrt_schedule_with_warmup(optimizer, self.config.common.mode.warmup_steps, self.config.common.mode.num_epochs)
        else:
            lr_schedule = None
      
        return optimizer, scheduler, lr_schedule


    def set_device(self, data, ignoreList):

        if isinstance(data, list):
            return [data[i].to(self.config.common.device).float() if i not in ignoreList else data[i] for i in range(len(data))]
        else:
            raise Exception('set device for input not defined')
            
            
    
