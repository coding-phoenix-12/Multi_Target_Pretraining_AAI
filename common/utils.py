from easydict import EasyDict
import yaml
import os
import torch
import numpy as np
from tqdm import tqdm
# from trainer import tts_trainer, ema_trainer, pta_tts_trainer
from trainer import trainer

# from models import fastspeech, fastspeech_with_ema, fastspeech_ptatts
from models import fastspeech, hier, pretrain, finetune, baseline

import librosa
from pathlib import Path
# from pyannote.audio import Model
# from pyannote.audio import Inference
import scipy.io
import soundfile as sf

def read_yaml(yamlFile):
    with open(yamlFile) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        cfg = EasyDict(config)
    return cfg

def t_(dataset):
    return torch.from_numpy(np.array(dataset))


def get_trainer(config):
    if config.common.mode.name == 'aai':
        return trainer
    else:
        raise NotImplementedError

def get_model(config):
    
    mode = config.common.mode.name
    subs = config.common.mode.num_speakers
    if config.common.feats in ['mfcc',]:
        input_dim = 13
    elif config.common.mode.feats in ['pase_plus',]:
        input_dim = 256
    elif config.common.feats in ['audio_albert', 'tera', 'mockingjay']:
        input_dim = 768 
    elif config.common.feats in ['vq_wav2vec', 'wav2vec', 'apc', 'npc']:
        input_dim = 512
    elif config.common.feats in ['decoar']:
        input_dim = 2048
    else:
        raise NotImplementedError
    
    if config.common.mode.model_size == 'default':
            model_config = read_yaml('config/fs.yaml')
    elif config.common.mode.model_size == 'small':
            model_config = read_yaml('config/fs_small.yaml')
    elif config.common.mode.model_size == 'large':
            model_config = read_yaml('config/fs_large.yaml')
            
     

    if "ema" in config.common.predict:
        if "baseline" in config.logging.run_name:
            model = baseline.FastSpeech(n_mel_channels=12,
                                                tag=mode,
                                                input_dim=input_dim,
                                                padding_idx=config.data.phonPadValue,
                                                n_speakers=subs,
                                                conf=config, 
                                                **model_config).to(config.common.device)
            
        else:
            model = finetune.FastSpeechft(n_mel_channels=12,
                                                tag=mode,
                                                input_dim=input_dim,
                                                padding_idx=config.data.phonPadValue,
                                                n_speakers=subs,
                                                conf=config, 
                                                model_conf=model_config,
                                                **model_config).to(config.common.device)
        
        
    
    elif "place" in config.common.predict or "manner" in config.common.predict or "height" in config.common.predict or "backness" in config.common.predict or "phones" in config.common.predict or "weights" in config.common.predict:
        model = pretrain.FastSpeech(n_mel_channels=12,
                                            tag=mode,
                                            input_dim=input_dim,
                                            padding_idx=config.data.phonPadValue,
                                            n_speakers=subs,
                                            conf=config, 
                                            **model_config).to(config.common.device)
    
    else:
        model = fastspeech.FastSpeech(n_mel_channels=12,
                                            tag=mode,
                                            input_dim=input_dim,
                                            use_one_hot_spk_embed=config.common.mode.one_hot_sub_embed,
                                            pyannote_sub_embed=config.common.mode.pyannote_sub_embed,
                                            sub_embed_loc=config.common.mode.sub_embed_loc,
                                            padding_idx=config.data.phonPadValue,
                                            n_speakers=subs,
                                            conf=config, 
                                            **model_config).to(config.common.device)
        
        

    return model


def load_pretrained(config, model):
    model.load_state_dict(torch.load(config.common.ema_pretrained))
    return model

MAX_WAV_VALUE = 32768.0

def get_audio(sample, lengths):
    y_gen_tst = sample[:int(lengths[0])].T
    y_gen_tst = np.exp(y_gen_tst)
    S = librosa.feature.inverse.mel_to_stft(
            y_gen_tst,
            power=1,
            sr=22050,
            n_fft=1024,
            fmin=0,
            fmax=8000.0)
    audio = librosa.core.griffinlim(
            S,
            n_iter=32,
            hop_length=256,
            win_length=1024)
    audio = audio * MAX_WAV_VALUE
    audio = audio.astype('int16')
    return audio

def find_audio(name, folder):
    all_files = set(get_files(folder))
    filename  = [str(f) for f in all_files if name in str(f)    ]
    assert len(filename) == 1
    return filename[0]

def get_files(path, extension='.wav'):
    if isinstance(path, str): path = Path(path).expanduser().resolve()
    return list(path.rglob(f'*{extension}'))

def support_aditional_feats(cfg):
    for feat in cfg.common.additional_feats:
        
        if feat == 'segment':
            get_start_stop_wav(cfg)
        elif feat == 'pyannote':
            extract_pyannote_feats(cfg)
        else:
            raise NotImplementedError

def get_start_stop_wav(cfg):
    F_all = {}
    begin_end_all = {}
    all_subs = os.listdir(os.path.join(cfg.data.rootPath, 'DataBase'))
    for spk_idx, sub in enumerate(sorted(all_subs)):
        F_all[sub] = {}
        for idx, wavfile in enumerate(sorted(os.listdir(os.path.join(cfg.data.rootPath, 'DataBase', sub, cfg.data.wavFolder)))):
            F_all[sub][Path(wavfile).stem] = idx
        begin_end_path = os.path.join(cfg.data.startStopFolder, sub)
        begin_end = scipy.io.loadmat(os.path.join(begin_end_path, os.listdir(begin_end_path)[0]))
        begin_end_all[sub] = begin_end['BGEN']
    wavfiles=get_files(os.path.join(cfg.data.rootPath, 'DataBase'))
    save_path = cfg.data.segment_path
    if os.path.exists(save_path):
        num_files = len(get_files(save_path))
        if num_files == len(wavfiles):
            return
    for filename in tqdm(wavfiles):
        subject = str(filename).split('/')[-4]
        folder = os.path.join(cfg.data.segment_path, subject)
        if not os.path.exists(folder): os.makedirs(folder)
        filestem = filename.stem
        F = F_all[subject][filestem]
        beginEnd = begin_end_all[subject]
        begin = beginEnd[0, F]
        end = beginEnd[1, F]
        y, sr = librosa.load(str(filename), sr=cfg.data.sampleRate)
        y = y[int(begin*sr):int(end*sr)]
        path = os.path.join(folder, filestem+'.wav')
        sf.write(path, y, sr)
    exit()
    
def extract_pyannote_feats(cfg):
    wavfiles = get_files(os.path.join(cfg.data.segment_path))
    
    with open(cfg.data.hf_token_path, 'r') as f:
        token = f.read().strip('\n')
    model = Model.from_pretrained("pyannote/embedding", 
                              use_auth_token=token)
    inference = Inference(model, window="whole")
    num_files = len(get_files(cfg.data.pyannote_path, '.pt'))
    if len(wavfiles) == num_files: return
    for filename in tqdm(wavfiles):
        subject = str(filename).split('/')[-2]
        embedding = inference(filename)
        save_path = os.path.join(cfg.data.pyannote_path, subject)
        if not os.path.exists(save_path): os.makedirs(save_path)
        save_path = os.path.join(save_path, filename.stem+'.pt')
        torch.save(embedding, save_path)
    exit()