
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from transformers import AutoModelForCausalLM
from typing import Optional, Dict, Callable
from peft import LoraConfig, get_peft_model


class TextDecoder(nn.Module):
    def __init__(
            self, 
            textdec_name: str, 
            **kwargs
    ) -> None:
        super().__init__()
        self.textdec_name = textdec_name.lower()
        self.ignore_index = -100
        self.lm = AutoModelForCausalLM.from_pretrained(textdec_name)
        if "gpt" in self.textdec_name: 
            self.sep_token_id = 50256
            self.pad_token_id = None  # !TODO add pad token id for gpt2
            self.embed_fn = self.lm.transformer.wte,
        elif "smollm2" in self.textdec_name: 
            self.sep_token_id = 0
            self.pad_token_id = 17 
            self.embed_fn = self.lm.model.embed_tokens
        else: 
            raise ValueError(f"Unsupported text decoder: {self.textdec_name}")

        if kwargs.get("use_lora", False) == True: 
            lora_config = LoraConfig(**kwargs["lora_config"])
            self.lm = get_peft_model(self.lm, lora_config)
            self.use_lora = True


if __name__ == "__main__": 
    import yaml
    with open("src/config/pretrain_m2t_htsat.yaml", "r") as f:  
        config = yaml.safe_load(f)
    text_decoder = TextDecoder(**config["model"]["decoder"])
    for n, p in text_decoder.named_parameters(): 
        print(f"{n}: {p.requires_grad}")
        if "lora" in n: 
            print(p.shape, '\n')

    # torch.save(text_decoder.state_dict(), "test/text_decoder.pt")

    # text_decoder.load_state_dict(torch.load("test/text_decoder.pt"))
    # print("Load success")
