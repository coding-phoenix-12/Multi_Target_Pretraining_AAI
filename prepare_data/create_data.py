import numpy as np
import torch
import librosa
import os
from tqdm import tqdm
from pqdm.processes import pqdm
from s3prl.nn import S3PRLUpstream


class DataPreprocessor:
    def __init__(self, conf):
        ph = torch.load("ph_map")
        self.ph_to_id = ph["ph_to_id"]
        self.id_to_ph = ph["id_to_ph"]
        self.wavs_path = "<wavs_path>"
        self.fa_path = "<forced alignemnt path>"
        self.save_path = "<save_path>/"
        self.model = S3PRLUpstream("tera")
        self.model.eval()
        

    def create_mfcc(self, y, sr, fstart, fend):
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, win_length=320, hop_length=160).T
        mfcc = mfcc[int(fstart):int(fend), :]
        return mfcc

    def get_fa_list(self, args):
        fa_list, wav = args["fa"], args["wav"]
        save_name = self.save_path + "libri360_" + wav.split("/")[-1].split(".")[0] + ".pt"
        if os.path.exists(save_name):
            return
        

        y, sr = librosa.load(wav, sr=16000)
        if fa_list[0].split()[-1].strip() == "SIL" or fa_list[0].split()[-1].strip() == "SPN":
            # y = y[(int(float(fa_list[0].split()[3].strip())) * 16000):]
            fa_list = fa_list[1:]
        if fa_list[-1].split()[-1].strip() == "SIL" or fa_list[-1].split()[-1].strip() == "SPN":
            # y = y[:int(float(fa_list[-1].split()[2].strip()) * 16000)]
            fa_list = fa_list[:-1]
            
        fstart = float(fa_list[0].split()[2].strip()) * 100
        fend = (float(fa_list[-1].split()[2].strip()) + float(fa_list[-1].split()[3].strip())) * 100
        # print(fstart, fend)



        mfcc = self.create_mfcc(y, sr, fstart, fend)
        err = 0
        phone_align = []
        phone_ids_align = []
        phones = []
        phone_ids = []
        t_sum = 0
        # prev_end = float(fa_list[0].split()[2].strip()) * 1000 

        # print("fa: ", len(fa_list))
        for i in range(len(fa_list)):
            start = float(fa_list[i].split()[2].strip()) * 16000
            end = float(fa_list[i].split()[3].strip()) * 16000 + start
            phone = fa_list[i].split()[-1].strip()
            if "_" in phone:
                phone = phone.split("_")[0]
                if phone[-1] in [str(x) for x in range(10)]:
                    phone = phone[:-1]
                    
            if phone == "SPN":
                phone = "SIL"
                    
            phone = phone.lower()
            dur = float(fa_list[i].split()[3].strip()) * 16000
            frames = int(dur // 160)
            
            rem = dur % 160
            err += rem

           
                
            # prev_end = start + dur
            

            if err >= 160:
                frames += 1
                err -= 160
                
            phone_id = self.ph_to_id[phone]
            phones.append(phone)
            phone_ids.append(phone_id)
            phone_align += [phone] * frames
            phone_ids_align += [phone_id] * frames
            # print(frames, dur, rem)
            t_sum += dur / 16000
            
            
        if len(phone_align) != mfcc.shape[0]:
            print("Error: ", wav.split("/")[-1], len(phone_align), mfcc.shape[0])
            phone_align = phone_align[:mfcc.shape[0]]
            phone_ids_align = phone_ids_align[:mfcc.shape[0]]


        
        torch.save({"mfcc": mfcc, "phone_align": phone_align, "phone_ids_align": phone_ids_align, "phones": phones, "phone_ids": phone_ids}, save_name)
        
        # print("align: ", len(phone_align))
        # print("mfcc: ", mfcc.shape, t_sum)
        # print("Error: ", err)
        # print(float(fa_list[0].split()[2].strip()), float(fa_list[-1].split()[2].strip()) + float(fa_list[-1].split()[3].strip()))
        # # print(phone_align)
        # exit(0)
        
        
    def clip_audio(self, args):
        fa_list, wav = args["fa"], args["wav"]
        

        y, sr = librosa.load(wav, sr=16000)
        if fa_list[0].split()[-1].strip() == "SIL" or fa_list[0].split()[-1].strip() == "SPN":
            # y = y[(int(float(fa_list[0].split()[3].strip())) * 16000):]
            fa_list = fa_list[1:]
        if fa_list[-1].split()[-1].strip() == "SIL" or fa_list[-1].split()[-1].strip() == "SPN":
            # y = y[:int(float(fa_list[-1].split()[2].strip()) * 16000)]
            fa_list = fa_list[:-1]
            
        fstart = float(fa_list[0].split()[2].strip()) * 100
        fend = (float(fa_list[-1].split()[2].strip()) + float(fa_list[-1].split()[3].strip())) * 100
        return y, sr, fstart, fend
        
        
        
        
        
    def generate_s3prl(self, wav, y, fstart, fend):
        tera_save_path = "<tera_save_path>/"
        # proc_file = wav.replace("audios", "processed").replace(".wav", ".pt")
        proc_file = self.save_path + "/libri_" + wav.split("/")[-1].split(".")[0] + ".pt"

        d = torch.load(proc_file)
        shape = len(d["phone_align"])
        # start, end = d["begin_end"][0], d["begin_end"][1]
        # # print(y.shape, start*256, end*256)
        # y = y[int(start*160):int(end*160)]
        
        y = torch.from_numpy(y).unsqueeze(0)
        all_hs, all_hs_len = self.model(y, torch.LongTensor([y.shape[1]]))
        last_layer = all_hs[-1].detach().numpy()[0]
        save_name = tera_save_path + "libri_"+ wav.split("/")[-1].split(".")[0] + ".pt"
        # print(last_layer.shape)
        # print(shape, last_layer.shape)
        last_layer = last_layer[int(fstart):int(fend), :]
        # print(last_layer.shape)
        torch.save(last_layer, save_name)
        
        
    def run(self):
        all_wavs = []
        for path, subdirs, files in os.walk(self.wavs_path):
            for name in files:
                if name.endswith(".flac"):
                    all_wavs.append(os.path.join(path, name))
                    
        with open(self.fa_path, "r") as f:
            fa = f.readlines()
            
        fa_files =  list(set([x.split()[0].strip() for x in fa]))
        all_wav_names = {x.split("/")[-1].split(".")[0]: x for x in all_wavs}
        # print(len(fa_files))
        # common_files = [x for x in all_wavs if x.split("/")[-1].split(".")[0] in fa_files]
        # full_fa_list = []
        full_wavs = []
        full_fa_dict = {}
        
        
        
        for fa_lines in tqdm(fa):
            fname = fa_lines.split()[0].strip()
            if fname not in all_wav_names.keys():
                continue
            if fname not in full_fa_dict:
                full_fa_dict[fname] = {
                    "fa": [],
                    "wav": all_wav_names[fname]
                }
            full_fa_dict[fname]["fa"].append(fa_lines)
        
        
        print(len(full_fa_dict))
        
        for fa_list_key in tqdm(full_fa_dict.keys()):
            self.get_fa_list(full_fa_dict[fa_list_key])
            y, sr, fstart, fend  = self.clip_audio(full_fa_dict[fa_list_key])
            # self.generate_s3prl(fa_list_key, y, fstart, fend)
            
            
            
            
            
            
            
if __name__ == "__main__":
    conf = None
    dp = DataPreprocessor(conf)
    dp.run()
        
        
                
        
            
            
            
        
        
        
        
        