import torch
import os
from train_accelerate import get_model_and_tokenizer
import warnings
warnings.filterwarnings("ignore")  
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--audio_path", required=True, type=str,
                        help="Path to audio file")
    parser.add_argument("-c", "--exp_dir", type=str, required=True, 
                        help="Exp dir to store model ckpt and config")
    parser.add_argument("-p", "--prompt", type=str,
                        default="Describe the audio you hear. ")
    parser.add_argument("-s", "--strategy", choices=["greedy", "top-p"], default="greedy")
    args = parser.parse_args()  

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_path = os.path.join(args.exp_dir, "models/config.yaml")
    model_ckpt_path = os.path.join(args.exp_dir, "models/best_step/best_model.pt")
    model, tokenizer, config = get_model_and_tokenizer(
        config=config_path, 
        model_ckpt_path=model_ckpt_path
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