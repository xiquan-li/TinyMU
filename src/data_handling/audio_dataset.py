import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from .dataset_utils import load_audio   # import the module relatively (from data_handling/ dir)
import numpy as np
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Gain, PolarityInversion
from typing import Optional, Dict, Tuple, List
import random


class AudioDataset(Dataset):
    def __init__(self, 
                 tokenizer: object, 
                 json_files: List[str], 
                 audio_dirs: List[str], 
                 sample_rate: int=32000, 
                 max_length: int=10,
                 max_text_token_len: int=129, 
                 wav_aug: bool=False, 
                 max_data_num: int=-1,
                 **kwargs):

        self.sample_rate = sample_rate
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.max_text_token_len = max_text_token_len
        self.data = []
        self.dataset_to_audio_dir = {}  # dataset to audio_dir mapping
        self.wav_aug = wav_aug
        if self.wav_aug: 
            self.augment = Compose([
                PolarityInversion(p=0.3),   
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.3),                           
                Gain(min_gain_db=-6, max_gain_db=6, p=0.5),              
                TimeStretch(min_rate=0.95, max_rate=1.05, p=0.5),           
                PitchShift(min_semitones=-2, max_semitones=2, p=0.5),      
            ])

        for idx, file in enumerate(json_files): 
            with open(file, "r") as f: 
                data = json.load(f)
            self.data += data
            dataset = data[0]['dataset']
            self.dataset_to_audio_dir[dataset] = audio_dirs[idx]
        
        print(f'Total data: {len(self.data)}, shuffling ...')
        random.shuffle(self.data)
        print(f'Shuffled data: {len(self.data)}')
        if max_data_num > 0:
            self.data = self.data[:max_data_num]
            print(f'Truncated data: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dataset = self.data[idx]['dataset']
        audio_dir = self.dataset_to_audio_dir[dataset]
        audio_path = os.path.join(audio_dir, self.data[idx]['file_name'])
        waveform = load_audio(audio_path, 
                              sample_rate=self.sample_rate, 
                              max_length=self.max_length,
                              random_crop=self.wav_aug) 
        if self.wav_aug: 
            waveform = self.augment(samples=waveform.numpy(), sample_rate=self.sample_rate)
            waveform = torch.from_numpy(waveform) 

        input_text = self.data[idx]['input_text']
        target_text = self.data[idx]['target_text']

        input_text_tok = self.tokenizer(
            text=input_text, 
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_text_token_len, 
            pad_to_max_length=True, 
            return_tensors="pt"
        )
        
        target_text_tok = self.tokenizer(
            text=target_text, 
            add_special_tokens=True,
        )  # avoid padding & truncation here 

        input_ids, input_mask = input_text_tok["input_ids"], \
            input_text_tok["attention_mask"]
        target_ids, target_mask = target_text_tok["input_ids"], \
            target_text_tok["attention_mask"]

        if len(target_ids) < self.max_text_token_len: 
            target_ids.append(self.tokenizer.eos_token_id) 
            target_mask.append(1)
            # manual padding 
            target_ids += [17]*(self.max_text_token_len-len(target_ids))
            target_mask += [0]*(self.max_text_token_len-len(target_mask))
        else: 
            # manual truncation
            target_ids = target_ids[:self.max_text_token_len]
            target_mask = target_mask[:self.max_text_token_len]

        target_ids = torch.tensor(
            target_ids, dtype=input_ids.dtype
        ).unsqueeze(0)
        target_mask = torch.tensor(
            target_mask, dtype=input_mask.dtype
        ).unsqueeze(0)

        return waveform, input_ids, input_mask, target_ids, target_mask, input_text, target_text
    

def collate_fn(batch): 

    def pad_batch(sequences, padding_value=0):
        return pad_sequence(
            sequences, 
            batch_first=True, 
            padding_value=padding_value
        )

    waveforms, input_ids, input_mask, target_ids, target_mask, input_text, target_text = \
        zip(*batch)

    waveforms_padded = pad_batch(waveforms, padding_value=0.0)
    waveform_lengths = [len(w) for w in waveforms]
    input_ids_padded = pad_batch(input_ids).squeeze()
    input_mask_padded = pad_batch(input_mask).squeeze()
    target_ids_padded = pad_batch(target_ids).squeeze()
    target_mask_padded = pad_batch(target_mask).squeeze()

    return {
        "waveforms": waveforms_padded,  
        "waveform_lengths": waveform_lengths, 
        "input_ids": input_ids_padded,  
        "input_mask": input_mask_padded,  
        "target_ids": target_ids_padded,  
        "target_mask": target_mask_padded,  
        "input_text": input_text,  
        "target_text": target_text,  
    }