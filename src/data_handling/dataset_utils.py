import torch
import os
import torchaudio
from torchaudio.transforms import Resample
import numpy as np


def load_audio(
        wav_path, 
        max_length: int = 10, 
        sample_rate: int = 32000, 
        random_crop: bool = False
) -> torch.Tensor: 
    try: 
        wav_info = torchaudio.info(wav_path)
        sr = wav_info.sample_rate  # sr of the audio 
        total_samples = wav_info.num_frames
        target_samples = int(max_length * sr)
        if not random_crop: 
            waveform, sr = torchaudio.load(
                wav_path, 
                num_frames=target_samples
            )   #[1, target_samples]
        else: 
            max_start = max(0, total_samples - target_samples)
            start = torch.randint(0, max_start + 1, ()).item() 
            waveform, sr = torchaudio.load(
                wav_path,
                frame_offset=start,
                num_frames=target_samples
            )  # [1, target_samples]

    except (FileNotFoundError, RuntimeError) as e: 
        waveform = torch.zeros((1, 32000))
        sr = 16000  
        print(f"Wav file: {wav_path} not found")

    if waveform.size(-1) < 0.1*sample_rate: 
        waveform = torch.zeros(max_length*sample_rate)
    else: 
        waveform = waveform[0]
        resampler = Resample(orig_freq=sr, new_freq=sample_rate)  # 32k for HTSAT, 16k for others
        waveform = resampler(waveform)  # (T,)
    
    return waveform

def pad_sequence(data):
    if isinstance(data[0], np.ndarray):
        data = [torch.as_tensor(arr) for arr in data]
    padded_seq = torch.nn.utils.rnn.pad_sequence(data,
                                                 batch_first=True)
    length = [x.size(0) for x in data]
    return padded_seq, length