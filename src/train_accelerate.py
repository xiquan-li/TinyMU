import warnings
warnings.filterwarnings("ignore") 
import time
from pprint import PrettyPrinter
import wandb
import torch
import argparse
import yaml
from tqdm import tqdm
from loguru import logger
from torch.utils.data import DataLoader
from data_handling.audio_dataset import AudioDataset, collate_fn
from utils import (
    setup_seed,
    seed_worker,
    AverageMeter,
    get_optimizer, 
    cosine_lr,
    constant_lr
)

from pathlib import Path
import sys
import warnings
import os
from models.m2t import m2t
from transformers import AutoTokenizer
from accelerate import Accelerator, DistributedDataParallelKwargs


def read_config_to_dict(config_path):
    return_dict = {}
    with open(config_path, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        return_dict[k] = v
    return return_dict


def get_model_and_tokenizer(config, model_ckpt_path=None, logger=None):
    r"""Load m2t with config file"""
    log_func = logger.info if logger else print
    if type(config) == str:  # read config from path
        config = read_config_to_dict(config_path=config)
    model = m2t(config["model"])
    
    ## loading pre-trained model
    audio_encoder_name = config["model"]["encoder"]["audioenc_name"].lower()
    if  audio_encoder_name == "htsat": 
        audio_encoder_path = "./weights/HTSAT.ckpt"
        log_func(f"Loading HTSAT audio encoder weights from {audio_encoder_path}")
        encoder_ckpt = torch.load(
            audio_encoder_path, map_location=torch.device('cpu')
        )["state_dict"]
        for key in list(encoder_ckpt.keys()):
            if key.startswith('sed_model'):
                v = encoder_ckpt.pop(key)  
                encoder_ckpt[key.replace("sed_model.", "htsat.")] = v  # remove 'sed_model.'
        out = model.audio_encoder.enc.load_state_dict( 
            encoder_ckpt, strict=False  # not to load c2l.weight and c2l.bias
        )
        log_func(f"Missing keys: {out.missing_keys}, unexpected keys: {out.unexpected_keys}")
    else: 
        log_func(f"Using audio encoder {audio_encoder_name} ... ")

    if model_ckpt_path != None: 
        log_func(f"Loading pre-trained model parameters from {model_ckpt_path}")
        model_state_dict = torch.load(model_ckpt_path, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(model_state_dict["model"])
    else: 
        log_func(f"We are NOT loading pre-trained m2t model weights")
    
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["decoder"]["textdec_name"])
    tokenizer.add_special_tokens({'pad_token': '!'})

    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer, config


def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config/finetune_m2t_htsat.yaml", type=str,
                        help="Model config files")
    parser.add_argument("-n", "--exp_name", default="exp_name", type=str,
                        help="Name of this experiment.")
    parser.add_argument("-l", "--lr", default=5e-5, type=float,
                        help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed")
    parser.add_argument("--btz", type=int, default=8, 
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=15, 
                        help="Number of epochs for training")
    parser.add_argument("--use_wandb", action="store_true", 
                        help="Whether or not use wandb")
    parser.add_argument("--pretrain_ckpt_path", type=str, required=False,
                        help="Pretrained m2t ckpt path for fine-tuning")
    args = parser.parse_args()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs]
    )
    num_gpus = accelerator.num_processes 

    ## logging
    exp_name = args.exp_name
    log_output_dir = Path(exp_name, 'logging')
    model_output_dir = Path(exp_name, 'models')
    exp_name = exp_name.split("/")[-1]
    log_output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process: 
        logger.remove()
        logger.add(sys.stdout, format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
                filter=lambda record: record['extra']['indent'] == 1)
        logger.add(log_output_dir.joinpath('training_log.txt'), format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
                filter=lambda record: record['extra']['indent'] == 1)
        main_logger = logger.bind(indent=1)
        printer = PrettyPrinter()
    else: 
        main_logger = None

    ## override config by cli
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config["optim_args"]["lr"] = args.lr
    config["training"]["seed"] = args.seed
    config["training"]["epochs"] = args.epochs  
    config["data"]["batch_size"] = args.btz
    device = torch.device(config["training"]["device"])
    # seed = config["training"]["seed"] + get_rank()
    setup_seed(args.seed)
    if args.use_wandb and accelerator.is_main_process: 
        wandb.init(
            project="m2t",
            name=exp_name,
            config=config
        )  # init wandb after updating the config file
    config_out_path = os.path.join(model_output_dir, "config.yaml")  
    if accelerator.is_main_process: 
        main_logger.info(f"Exp name: {exp_name}")
        with open(config_out_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    
    ## initialize model & freezing
    model, tokenizer, _ = get_model_and_tokenizer(config, args.pretrain_ckpt_path, logger=main_logger)
    if args.use_wandb and accelerator.is_main_process: 
        wandb.watch(model)

    if config["model"]["encoder"]["freeze"] == True: 
        if accelerator.is_main_process: 
            main_logger.info(f"Freezing audio encoder ...")
        for name, param in model.audio_encoder.enc.named_parameters(): 
            if model.audio_encoder.audioenc_name == "htsat":  
                if name.startswith("htsat"):  # Only freezing the htsat part of the audio encoder
                    param.requires_grad = False
            else:
                param.requires_grad = False
    if config["model"]["decoder"]["freeze"] == True: 
        if config["model"]["decoder"].get("use_lora", False) == True: 
            raise ValueError("Please set NOT freeze lm when using lora")
        if accelerator.is_main_process: 
            main_logger.info(f"Freezing text decoder ...")
        for name, param in model.text_decoder.named_parameters():
            param.requires_grad = False

    ## prepare training 
    if config["model"]["encoder"]["audioenc_name"].lower() == "htsat": 
        config["data"]["sample_rate"] = 32000
    elif config["model"]["encoder"]["audioenc_name"].lower() == "matpac": 
        config["data"]["sample_rate"] = 16000
    else: 
        raise ValueError(f"Audio encoder {config['model']['encoder']['audioenc_name']} is not supported")

    train_dataset = AudioDataset(tokenizer=tokenizer, 
                                 json_files=config["data"]["train_json_files"], 
                                 audio_dirs=config["data"]["train_audio_dirs"], 
                                 sample_rate=config["data"]["sample_rate"], 
                                 max_length=config["data"]["max_length"],   # max audio len (in sec)
                                 wav_aug=config["data"]["wav_aug"],
                                 max_text_token_len=config['data']['max_text_token_len'],
                                 max_data_num=config["data"]["max_data_num"])
    dataloader = DataLoader(dataset=train_dataset,
                            collate_fn=collate_fn,
                            shuffle=True,
                            batch_size=config["data"]["batch_size"],
                            num_workers=config["data"]["num_workers"],
                            worker_init_fn=seed_worker)
    optimizer = get_optimizer(model.parameters(),
                              lr=config["optim_args"]["lr"],
                              betas=config["optim_args"]["betas"],
                              eps=config["optim_args"]["eps"],
                              momentum=config["optim_args"]["momentum"],
                              optimizer_name=config["optim_args"]["optimizer_name"])
    total_steps = len(dataloader) * config["training"]["epochs"] // num_gpus

    if config["optim_args"]["scheduler"] == "cosine": 
        scheduler = cosine_lr(optimizer,
                            base_lr=config["optim_args"]["lr"],
                            warmup_length=int(total_steps*config["optim_args"]["warmup_ratio"]),
                            total_steps=total_steps) 
    elif config["optim_args"]["scheduler"] == "constant": 
        scheduler = constant_lr(optimizer,
                            base_lr=config["optim_args"]["lr"],
                            warmup_length=int(total_steps*config["optim_args"]["warmup_ratio"])) 
    
    ## use accelerator for ddp training
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)  # close detection
    start_epoch = 1
    max_epoch = config["training"]["epochs"]
    if accelerator.is_main_process: 
        main_logger.info('Training setting:\n'
                        f'{printer.pformat(config)}')
        model_without_ddp = accelerator.unwrap_model(model) 
        main_logger.info(f'Total number of parameters: {sum([i.numel() for i in model_without_ddp.parameters()])}')
        main_logger.info(f'Total number of trainable parameters: {sum([i.numel() for i in model_without_ddp.parameters() if i.requires_grad==True])}')
        main_logger.info(f'Trainable parameters of audio encoder: {sum([i.numel() for i in model_without_ddp.audio_encoder.parameters() if i.requires_grad==True])}')
        main_logger.info(f'Trainable parameters of projector: {sum([i.numel() for i in model_without_ddp.projector.parameters() if i.requires_grad==True])}')
        main_logger.info(f'Trainable parameters of text decoder: {sum([i.numel() for i in model_without_ddp.text_decoder.parameters() if i.requires_grad==True])}')
        main_logger.info('We are training {ratio:.2%} parameters of the text decoder'.format(ratio=sum([i.numel() for i in model_without_ddp.text_decoder.parameters() if i.requires_grad==True])
                                                                                              /sum([i.numel() for i in model_without_ddp.text_decoder.parameters()])))
        main_logger.info(f'Size of training set: {len(dataloader.dataset)}, number of iterations per epoch: {len(dataloader)}')

    ## prepare validation & evaluation 
    val_dataset = AudioDataset(tokenizer=tokenizer, 
                               json_files=config["data"]["val_json_files"], 
                               audio_dirs=config["data"]["val_audio_dirs"], 
                               sample_rate=config["data"]["sample_rate"], 
                               max_length=config["data"]["max_length"],  # max audio len (in sec)
                               wav_aug=config["data"]["wav_aug"], 
                               max_text_token_len=config['data']['max_text_token_len'])
    val_dataloader = DataLoader(dataset=val_dataset,
                                collate_fn=collate_fn,
                                shuffle=False,
                                batch_size=config["data"]["batch_size"],
                                num_workers=config["data"]["num_workers"],
                                drop_last=True,
                                worker_init_fn=seed_worker)
    val_dataloader = accelerator.prepare(val_dataloader)
    validation_step = config["training"]["validation_step"]

    ## training
    min_val_loss = torch.inf
    best_step = 0
    for epoch in range(start_epoch, max_epoch + 1):
        if accelerator.is_main_process: 
            main_logger.info(f'Training for epoch [{epoch}]')

        model.train()
        epoch_loss = AverageMeter()
        step_loss = AverageMeter()
        start_time = accelerator.gather(         
            torch.tensor(time.time())          
                .to(accelerator.device)       
        ).min().item()  # gather all the timestamps to track 
        if accelerator.is_main_process: 
            pbar = tqdm(colour='blue', total=len(dataloader), desc=f"Training Epoch {epoch}", dynamic_ncols=True)

        for batch_id, batch in enumerate(dataloader):

            optimizer.zero_grad()
            step = len(dataloader) * (epoch - 1) + batch_id
            scheduler(step)  # update lr
            if args.use_wandb and accelerator.is_main_process: 
                wandb.log({"train/lr": optimizer.param_groups[0]["lr"]})

            for k, v in batch.items(): 
                if isinstance(v, torch.Tensor): 
                    batch[k] = v.to(device, non_blocking=True)      
            out = model(**batch)
            loss = out.loss
            accelerator.backward(loss)
            gathered_loss = accelerator.gather(loss.detach()).mean()
            epoch_loss.update(gathered_loss.item())
            step_loss.update(gathered_loss.item())
            optimizer.step()

            if accelerator.is_main_process:  
                pbar.update(1) 
                if args.use_wandb: 
                    wandb.log({"train/loss": gathered_loss.item()})

            if (step+1) % validation_step == 0: 
                with torch.no_grad(): 
                    if accelerator.is_main_process: 
                        main_logger.info(f"Validation for step [{step}]")
                    val_statics = validate(model, val_dataloader, device, args.use_wandb, accelerator)
                    val_loss = val_statics["loss"]
                    elapsed_time = val_statics["time"]
                    if accelerator.is_main_process: 
                        main_logger.info(f"Validation statistics:\nloss for step [{step}]: {val_loss:.3f}",
                                        f"\ttime: {elapsed_time:.1f}")
                        if val_loss < min_val_loss: 
                            main_logger.info(f"Saving model for step {step}")
                            model_without_ddp = accelerator.unwrap_model(model) 
                            sav_obj = {
                                "model": model_without_ddp.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "config": config,
                                "epoch": epoch
                            }
                            save_dir = str(model_output_dir) + f"/best_step"  # FIX: only save the best model
                            if not os.path.exists(save_dir): 
                                os.makedirs(save_dir)
                            torch.save(sav_obj, f"{save_dir}/best_model.pt")  # TODO: considering only save trainable parts to save memory
                            min_val_loss = val_loss
                            best_step = step+1
                model.train()
                if args.use_wandb and accelerator.is_main_process:
                    wandb.log({"train/step_loss": step_loss.avg})

        end_time = accelerator.gather(         
            torch.tensor(time.time())          
                .to(accelerator.device)       
        ).min().item() 
        elapsed_time = end_time - start_time

        if args.use_wandb and accelerator.is_main_process:
            wandb.log({"train/epoch_loss": epoch_loss.avg,
                    "train/epoch": epoch})
        
        if accelerator.is_main_process: 
            main_logger.info(f'Training statistics:\tloss for epoch [{epoch}]: {epoch_loss.avg:.3f},'
                            f'\ttime: {elapsed_time:.1f}, lr: {optimizer.param_groups[0]["lr"]:.6f}.')
            if epoch == max_epoch: 
                main_logger.info(f"Saving model for last epoch: {epoch}")
                model_without_ddp = accelerator.unwrap_model(model) 
                sav_obj = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "epoch": epoch
                }
                save_dir = str(model_output_dir) + f"/last_epoch" 
                if not os.path.exists(save_dir): 
                    os.makedirs(save_dir)
                torch.save(sav_obj, f"{save_dir}/best_model.pt")  # TODO: considering only save trainable parts to save memory
            
    ## finish training
    if accelerator.is_main_process:
        main_logger.info("Finish training")
        if args.use_wandb:  
            wandb.finish()


@torch.no_grad()
def validate(model, dataloader, device, use_wandb, accelerator):
    model.eval()

    epoch_loss = AverageMeter()
    start_time = accelerator.gather(         
        torch.tensor(time.time()).to(accelerator.device)       
    ).min().item()  

    pbar = tqdm(colour='green', total=len(dataloader),desc=f"Validation", dynamic_ncols=True)
    for batch_id, batch in enumerate(dataloader):

        for k, v in batch.items(): 
            if isinstance(v, torch.Tensor): 
                batch[k] = v.to(device, non_blocking=True)      

        out = model(**batch)
        loss = out.loss
        gathered_loss = accelerator.gather(loss.detach()).mean()
        epoch_loss.update(gathered_loss.item())  # FIXME average according to btz on different GPUs
        if accelerator.is_main_process:  
            pbar.update(1) 

    end_time = accelerator.gather(         
        torch.tensor(time.time()).to(accelerator.device)       
    ).min().item()  
    elapsed_time = end_time - start_time

    if use_wandb and accelerator.is_main_process: 
        wandb.log({"val/loss": epoch_loss.avg})

    return {
        "loss": epoch_loss.avg,
        "time": elapsed_time
    }


if __name__ == '__main__':
    main()