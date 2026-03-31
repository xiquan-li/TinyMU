import torch
import os
import sys
sys.path.append('./src/')
from src.train_accelerate import get_model_and_tokenizer
import warnings
warnings.filterwarnings("ignore")  
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", default='./resource/example1.wav', type=str,
                        help="Path to audio file")
    parser.add_argument("--config_path", type=str, default='./ckpt/tinymu.yaml', 
                        help="Path to the config yaml")
    parser.add_argument("--ckpt_path", type=str, default='./ckpt/tinymu.pt', 
                        help="Path to the ckpt")
    parser.add_argument("--prompt", type=str,
                        default="Describe the audio you hear. ")
    parser.add_argument("--strategy", choices=["greedy", "top-p"], default="greedy")
    args = parser.parse_args()  

    if not os.path.exists(args.config_path) or not os.path.exists(args.ckpt_path):
        print(f'Model not found at {args.ckpt_path}')
        print('Downloading models to "./ckpt/"...')
        try:
            weights_dir = Path('./ckpt')
            weights_dir.mkdir(exist_ok=True)
            snapshot_download(repo_id="AndreasXi/TinyMU", local_dir="./ckpt" )
        except Exception as e:
            print(f"Failed to download model: {e}")
            raise FileNotFoundError(f"Model file not found and download failed: {args.ckpt_path}, you may need to download the model manually.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = get_model_and_tokenizer(
        config=args.config_path, 
        model_ckpt_path=args.ckpt_path
    )

    print("="*50, "Parameter Summary", "="*50)
    print(f"Audio Encoder: {sum(p.numel() for p in model.audio_encoder.parameters())/1e6:.2f}M")
    print(f"Projector: {sum(p.numel() for p in model.projector.parameters())/1e6:.2f}M")
    print(f"Text Decoder: {sum(p.numel() for p in model.text_decoder.parameters())/1e6:.2f}M")
    print(f"Total: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    
    samples = [(args.audio_path, args.prompt)] # defaults to first audio
    
    with torch.no_grad(): 
        response = model.generate(samples=samples, 
                                  max_len=300, 
                                  top_p=0.8, 
                                  temperature=1.0,
                                  tokenizer=tokenizer,
                                  strategy=args.strategy)  # FIXME modify the arguments here in the model.generate function
    
    print("="*50, "Generation results", "="*50)
    print(f"Input prompt: {args.prompt}")
    print(f"Audio path: {args.audio_path}")
    print(f"Model output: {response}")