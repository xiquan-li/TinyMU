import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from .decoder import TextDecoder
from .encoder import AudioEncoder
from .projector import get_projector
from data_handling.dataset_utils import load_audio, pad_sequence
from typing import Optional, Dict, Tuple, List


class m2t(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()    
        self.config = model_config    
        self.audio_encoder = AudioEncoder(**model_config["encoder"])
        self.projector = get_projector(**model_config["projector"])
        self.text_decoder = TextDecoder(**model_config["decoder"])
        self.max_text_token_len = model_config["decoder"]["max_text_token_len"]  # FIXME: avoid storing this variable here, instead using config to pass the values

    def forward(self,
            waveforms: torch.Tensor,
            input_ids: torch.Tensor, 
            input_mask: torch.Tensor, 
            target_ids: Optional[torch.Tensor] = None, 
            target_mask: Optional[torch.Tensor] = None,
            waveform_lengths: Optional[torch.Tensor] = None, 
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        # import ipdb; ipdb.set_trace()
        audio_embeds = self.audio_encoder(waveforms, waveform_lengths)  # downsampled
        audio_embeds = self.projector(audio_embeds).contiguous() # projected
        audio_mask = torch.ones(  
            audio_embeds.size(0), 
            audio_embeds.size(1),
            dtype=torch.int64, 
            device=audio_embeds.device
        )  

        sep_token = torch.tensor([self.text_decoder.sep_token_id]).to(audio_embeds.device)
        sep_embed = self.text_decoder.embed_fn(sep_token) \
            .unsqueeze(0) \
            .repeat(audio_embeds.shape[0], 1, 1) \
            .to(audio_embeds.device)
        sep_mask = torch.ones(
            sep_embed.size(0), 
            1, 
            dtype=torch.int64, 
            device=sep_embed.device
        )
        input_text_embeds = self.text_decoder.embed_fn(input_ids).contiguous()  # [audio, sep, prompt]
        prefix = torch.cat([  # [1, sep, prompt]
            audio_embeds, 
            sep_embed, 
            input_text_embeds
        ], dim=1)  
        prefix_mask = torch.cat([audio_mask, sep_mask, input_mask], dim=1)  # [1, 1, input_text_mask]

        if kwargs.get("inference_mode", False) == True: 
            return prefix, prefix_mask  # early return for prefix generation during inference

        target_text_embeds = self.text_decoder.embed_fn(target_ids).contiguous()
        full_embeds = torch.cat([ 
            prefix, 
            target_text_embeds
        ], dim=1)   
        full_mask = torch.cat([prefix_mask, target_mask], dim=1)  # [1, 1, input_text_mask, target_text_mask]

        dummy_tokens = torch.full(
            size=(prefix.size(0), prefix.size(1)), 
            fill_value=self.text_decoder.pad_token_id, 
            dtype=torch.int64, 
            device=prefix.device
        ) 
        labels = torch.cat((dummy_tokens, target_ids), dim=1).to(prefix.device)
        labels[labels == self.text_decoder.pad_token_id] = self.text_decoder.ignore_index  # [-100, -100, -100, ans]
            
        out = self.text_decoder.lm(
            inputs_embeds=full_embeds, 
            labels=labels, 
            attention_mask=full_mask
        )
        return out


    def generate(
            self, 
            samples: List[Tuple[str, str]], 
            max_len: int, 
            top_p: float, 
            temperature: float, 
            tokenizer: object, 
            stop_token: str = '<|endoftext|>', 
            device: str = "cuda", 
            strategy: str = "greedy"
    ) -> List[str]:
        self.eval()
        audio_paths, prompts = [], []
        for sample in samples: 
            audio_paths.append(sample[0])
            prompts.append(sample[1])
        waveforms, waveforms_lengths = pad_sequence([
            load_audio(ap, sample_rate=self.audio_encoder.sr, random_crop=False)  # during infer we only use starting wav points
            for ap in audio_paths
        ])
        waveforms = waveforms.to(device)

        input_tok = tokenizer(
            text=prompts, 
            add_special_tokens=True,
            truncation=True,
            max_length=self.config["decoder"]["max_text_token_len"],
            pad_to_max_length=True, 
            return_tensors="pt"
        ).to(device)
        prefix, prefix_mask = self.forward(
            waveforms=waveforms,
            waveform_lengths=waveforms_lengths, 
            input_ids=input_tok["input_ids"],
            input_mask=input_tok["attention_mask"],
            inference_mode=True
        )
        preds = self._generate_batch(
            tokenizer=tokenizer, 
            embed=prefix, 
            top_p=top_p, 
            temperature=temperature, 
            stop_token=stop_token, 
            entry_length=max_len, 
            attention_mask=prefix_mask,
            strategy=strategy
        )
        return preds


    def _generate_batch(
            self,
            tokenizer,  # !TODO put tokenizer into m2t model
            embed: torch.Tensor=None,
            entry_length=300,  # maximum number of words
            top_p=0.8,
            temperature=1.,
            stop_token: str = '<|endoftext|>',
            attention_mask=None, 
            strategy="greedy"
        ):
        self.eval()
        tokens = None
        stop_token_index = tokenizer.encode(stop_token)[0]
        filter_value = -float("Inf")

        with torch.no_grad():
            if embed is not None:
                generated_embed = embed

            for i in tqdm(range(entry_length)):
                outputs = self.text_decoder.lm(inputs_embeds=generated_embed, attention_mask=attention_mask)   
                logits = outputs.logits   # (btz, seq, vocab)
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)  # (btz, vocab)
                sorted_logits, sorted_indices = torch.sort(
                    logits, 
                    descending=True,
                    dim=-1
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), 
                    dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone() 
                sorted_indices_to_remove[..., 0] = 0   # always keep the first (highest prob) logit

                for k in range(sorted_indices_to_remove.size(0)):
                    indices_to_remove = sorted_indices[k][sorted_indices_to_remove[k]]
                    logits[k, indices_to_remove] = filter_value

                if strategy == "greedy": 
                    next_token = torch.argmax(logits, -1).unsqueeze(1)
                elif strategy == "top-p": 
                    next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                next_token_embed = self.text_decoder.embed_fn(next_token)

                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated_embed = torch.cat([
                    generated_embed, 
                    next_token_embed], 
                dim=1)
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones_like(next_token)], 
                dim=1)

                condition = (tokens == stop_token_index).sum(dim=-1)
                if (condition > 0).all():
                    break
            
            output_list = list(tokens.squeeze().cpu().numpy())
            if output_list[0].ndim == 0:
                output_list = [output_list]
            generated_list = [tokenizer.decode(x).split("<|endoftext|>")[0] for x in output_list]

        return generated_list

    
if __name__ == "__main__": 
    # python -m src.models.m2t
    import yaml
    with open("src/config/pretrain_m2t_htsat.yaml", "r") as f: 
        config = yaml.safe_load(f)
    model = m2t(config["model"])

    print(f"Audio encoder count: { sum([p.numel() for p in model.audio_encoder.parameters()]) }")
    print(f"Audio encoder trainable count: { sum([p.numel() for p in model.audio_encoder.parameters() if p.requires_grad == True]) }\n")

    print(f"Projector count: { sum([p.numel() for p in model.projector.parameters()]) }")
    print(f"Projector trainable count: { sum([p.numel() for p in model.projector.parameters() if p.requires_grad == True]) }\n")

    print(f"Text decoder count: { sum([p.numel() for p in model.text_decoder.parameters()]) }")
    print(f"Text decoder trainable count: { sum([p.numel() for p in model.text_decoder.parameters() if p.requires_grad == True]) }")