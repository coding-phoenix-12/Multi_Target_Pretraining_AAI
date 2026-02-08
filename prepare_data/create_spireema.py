import numpy as np
import torch
import librosa
import os
from tqdm import tqdm
from pqdm.processes import pqdm
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
from s3prl.nn import S3PRLUpstream


class DataPreprocessor:
    def __init__(self, conf):
        ph = torch.load("ph_map", weights_only=False)
        self.ph_to_id = ph["ph_to_id"]
        self.id_to_ph = ph["id_to_ph"]
        self.root_path = "<SpireEMA datapath>"
        # self.model = S3PRLUpstream("tera")
        # self.model.eval()

        self.tera_model = S3PRLUpstream("tera")
        self.tera_model.eval()

        self.hubert_model = S3PRLUpstream("hubert")
        self.hubert_model.eval()

        self.w2v2_model = S3PRLUpstream("wav2vec2")
        self.w2v2_model.eval()

        self.save_path = "<save_path>/"
        
        
        
    def create_mfcc(self, y, sr, fstart, fend):
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, win_length=320, hop_length=160).T
        mfcc = mfcc[int(fstart):int(fend), :]
        return mfcc
    
    def generate_s3prl(self, wav):

        signal = torch.from_numpy(wav).unsqueeze(0)
        with torch.no_grad():
            tera_feats, _ = self.tera_model(signal, torch.tensor([wav.shape[0]]))
            tera_feats = tera_feats[-1].squeeze().numpy()

            hubert_feats, _ = self.hubert_model(signal, torch.tensor([wav.shape[0]]))
            hubert_feats = hubert_feats[-1].squeeze().numpy()

            w2v2_feats, _ = self.w2v2_model(signal, torch.tensor([wav.shape[0]]))
            w2v2_feats = w2v2_feats[-1].squeeze().numpy()

        return tera_feats, hubert_feats, w2v2_feats
    
    
    def get_mel(self, wav):
        signal = torch.from_numpy(wav)

        spectrogram, _ = mel_spectogram(
            audio=signal,
            sample_rate=16000,
            hop_length=256,
            win_length=1024,
            n_mels=80,
            n_fft=1024,
            f_min=0.0,
            f_max=8000.0,
            power=1,
            normalized=False,
            min_max_energy_norm=True,
            norm="slaney",
            mel_scale="slaney",
            compression=True
        )
        
        spectrogram = spectrogram.numpy()
        return spectrogram
    
    def get_alignments(self, phonemes, durations):
        phone_aligns = []
        phone_ids_aligns = []
        phones = []
        phone_ids = []
        
        for idx, phone in enumerate(phonemes):
            phone_id = self.ph_to_id[phone]
            phone_ids.append(phone_id)
            phones.append(phone)
            
            phone_aligns += [phone] * int(durations[idx])
            phone_ids_aligns += [phone_id] * int(durations[idx])
            
            
        return phone_aligns, phone_ids_aligns, phones, phone_ids
            
            
    
    def create_pt(self, wav):
        proc_file = wav.replace("audios", "processed").replace(".wav", ".pt")
        d = torch.load(proc_file, weights_only=False)
        # print(d.keys())
        wavs_path = "<wavs_path>/"
        y, sr = librosa.load(wav, sr=16000)
        mfcc = self.create_mfcc(y, sr, d["begin_end"][0], d["begin_end"][1])

        mel_y, _ = librosa.load(os.path.join(wavs_path, "SpireEMA_" + wav.split("/")[-2] + "_" + wav.split("/")[-1].split(".")[0] + "__" + wav.split("/")[-2] + ".wav"), sr=16000)
        mel_spectogram = self.get_mel(mel_y)

        phones = d["phonemes"]
        durs = d["durations"]
        phone_align, phone_ids_align, phones, phone_ids = self.get_alignments(phones, durs)
        ema_norm = d["ema_trimmed_and_normalised_with_6_articulators"]
        ema_raw = d["ema_raw"]
        ema_clipped = d["ema_trimmed"] 
        
        tera, hubert, w2v2 = self.generate_s3prl(y)

        tera = tera[d["begin_end"][0]: d["begin_end"][1], :]
        hubert = hubert[d["begin_end"][0]: d["begin_end"][1], :]
        w2v2 = w2v2[d["begin_end"][0]: d["begin_end"][1], :]

    
        save_name = self.save_path + "/SpireEMA_" + wav.split("/")[-2] + "_" + wav.split("/")[-1].split(".")[0] + ".pt"
        torch.save({"mfcc": mfcc, "phone_align": phone_align, "phone_ids_align": phone_ids_align, "phones": phones, "phone_ids": phone_ids, 
                    "ema_raw": ema_raw, "ema_norm": ema_norm, "ema_clipped": ema_clipped, "mel_spectogram": mel_spectogram, 
                    }, save_name)
        
        

        
    def run(self):
        all_wavs = []
        for root, dirs, files in os.walk(self.root_path + "audios"):
            for file in files:
                if file.endswith(".wav"):
                    all_wavs.append(os.path.join(root, file))
                    
        # pqdm(all_wavs, self.create_pt, n_jobs=8)
        for wav in tqdm(all_wavs):
            self.create_pt(wav)
        
        
if __name__ == "__main__":
    conf = None
    dp = DataPreprocessor(conf)
    dp.run()