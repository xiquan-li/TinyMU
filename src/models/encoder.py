import torch
import torch.nn as nn
from .htsat.htsat import HTSATWrapper
from transformers import AutoModel, Wav2Vec2FeatureExtractor
import torch.nn.functional as F


class AudioEncoder(nn.Module):
    def __init__(
            self, 
            audioenc_name: str, 
            ds_rate: int, 
            **kwargs
    ) -> None:
        super().__init__()
        self.audioenc_name = audioenc_name.lower()
        self.ds_rate = ds_rate
        self.config = kwargs
        self.sr = 32000 if self.audioenc_name == "htsat" else 16000  # for generation
        self.enc = self.get_audio_encoder()

    def forward(self, x, wav_lens):
        if self.audioenc_name == "htsat": 
            out_dict = self.enc(x)
            audio_embeds = out_dict['embedding']  # (btz, seq=1025, d)

        elif self.audioenc_name == "matpac": 
            audio_embeds, layer_results = self.enc(x)

        elif self.audioenc_name == "mert": 
            if not isinstance(wav_lens, torch.Tensor):
                wav_lens = torch.tensor(wav_lens, dtype=torch.long).to(x.device)
            max_length = wav_lens.max().item()  # FIXME use wav2vec feature extractor here
            range_tensor = torch.arange(
                max_length, 
                device=wav_lens.device
            ).expand(len(wav_lens), max_length)
            wav_mask = (range_tensor < wav_lens.unsqueeze(1)).float()
            outputs = self.enc(x, attention_mask=wav_mask, output_hidden_states=True)
            audio_embeds = outputs.last_hidden_state # (btz, seq=750, d)
        
        elif self.audioenc_name == "muq": 
            if not isinstance(wav_lens, torch.Tensor):
                wav_lens = torch.tensor(wav_lens, dtype=torch.long).to(audio_embeds.device)
            max_length = wav_lens.max().item()
            range_tensor = torch.arange(
                max_length, 
                device=wav_lens.device
            ).expand(len(wav_lens), max_length)
            wav_mask = (range_tensor < wav_lens.unsqueeze(1)).float()
            outputs = self.enc(x, attention_mask=wav_mask, output_hidden_states=True)
            audio_embeds = outputs.last_hidden_state  # (btz, seq=250, d)
            
        audio_embeds = self.downsample(audio_embeds)
        return audio_embeds

    def downsample(self, x):
        if self.audioenc_name == "htsat": 
            # x: (btz, seq, d) with first element being clip-level feature
            clip_latent = x[:,0,:].unsqueeze(1) 
            pooled = F.avg_pool2d(x[:,1:,:], kernel_size=(self.ds_rate,1))  
            x = torch.concat((clip_latent, pooled),axis=1)
        else: 
            # x: (btz, seq, d)
            x = F.avg_pool2d(x, kernel_size=(self.ds_rate,1))
        return x
    
    
    def get_audio_encoder(self):
        name = self.audioenc_name.lower()
        if name == "htsat":
            return HTSATWrapper(self.config["htsat_c2l_first"], self.config["htsat_wo_repeat"])

        elif name == 'matpac': 
            from .matpac.model import get_matpac
            ckpt_path = self.config.get("matpac_ckpt_path", None)
            model = get_matpac(checkpoint_path=ckpt_path, pull_time_dimension=False)
            return model

        elif name == "mert": 
            raise Exception('MERT is not supported yet')
            model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
            processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True)
            return model, processor

        elif name == "muq": 
            raise Exception('MuQ is not supported yet')
            from muq import MuQ
            return MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")

        else:
            raise Exception('The audio encoder name {} is incorrect or not supported'.format(name))

