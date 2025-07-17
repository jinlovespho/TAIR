import os 
import sys
import wandb 
import argparse
from omegaconf import OmegaConf

import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from terediff.model import ControlLDM, SwinIR, Diffusion
from terediff.utils.common import instantiate_from_config, to, log_txt_as_img
from terediff.dataset.codeformer import collate_fn_code
from terediff.dataset.realesrgan import collate_fn_real


def load_experiment_settings(accelerator, cfg):
    
    if cfg.exp_args.mode == 'TRAIN':
        datasets='_'.join((cfg.dataset.train.params.data_args['datasets']))
        exp_name = f"pho_server{cfg.log_args.server}_gpu{cfg.log_args.gpu}_{cfg.exp_args.mode}_{cfg.exp_args.model_name}_{cfg.exp_args.finetuning_method}_bs{cfg.train.batch_size}_lr{cfg.train.learning_rate}_{cfg.log_args.additional_msg}"

        # setup an experiment folder
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, exp_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        
    elif cfg.exp_args.mode == 'VAL':
        
        if cfg.prompter_args.use_vlm_prompt:
            exp_name=f'{cfg.log_args.user}_server{cfg.log_args.server}_gpu{cfg.log_args.gpu}_{cfg.dataset.val_dataset_name}_{cfg.exp_args.mode}_{cfg.exp_args.model_name}_infSampleStep{cfg.exp_args.inf_sample_step}_{cfg.vlm_args.vlm_name}_vlmInput{cfg.vlm_args.vlm_input_img}_vlmstep{cfg.vlm_args.inf_vlm_step}_vlmCorrection{cfg.vlm_args.vlm_text_correction}'
        else:
            exp_name=f'{cfg.log_args.user}_server{cfg.log_args.server}_gpu{cfg.log_args.gpu}_{cfg.dataset.val_dataset_name}_{cfg.exp_args.mode}_{cfg.exp_args.model_name}_infSampleStep{cfg.exp_args.inf_sample_step}'
        exp_dir=None
        ckpt_dir=None

    if accelerator.is_main_process:
        # setup logging tool
        if cfg.log_args.log_tool == 'wandb':
            wandb.login(key=cfg.log_args.wandb_key)
            wandb.init(project=cfg.log_args.wandb_proj_name, 
                    name=exp_name, 
                    config=argparse.Namespace(**OmegaConf.to_container(cfg, resolve=True))
            )
        return exp_dir, ckpt_dir, exp_name, None


def load_data(accelerator, cfg):

    # set dataset 
    train_ds = instantiate_from_config(cfg.dataset.train)
    val_ds = instantiate_from_config(cfg.dataset.val)
    
    if cfg.dataset.dataset_type == 'realsr':
        collate_fn = collate_fn_real

    # set data loader 
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=cfg.val.batch_size,
        num_workers=cfg.val.num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_ds, val_ds, train_loader, val_loader 



def load_model(accelerator, device, args, cfg):

    loaded_models={}

    # default: load cldm, swinir
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
    unused, missing = cldm.load_pretrained_sd(sd)
    if accelerator.is_main_process:
        print(
            f"strictly load pretrained SD weight from {cfg.train.sd_path}\n"
            f"unused weights: {unused}\n"
            f"missing weights: {missing}"
        )

    if cfg.train.resume:
        cldm.load_controlnet_from_ckpt(torch.load(cfg.train.resume, map_location="cpu"))
        if accelerator.is_main_process:
            print(
                f"strictly load controlnet weight from checkpoint: {cfg.train.resume}"
            )
    else:
        init_with_new_zero, init_with_scratch = cldm.load_controlnet_from_unet()
        if accelerator.is_main_process:
            print(
                f"strictly load controlnet weight from pretrained SD\n"
                f"weights initialized with newly added zeros: {init_with_new_zero}\n"
                f"weights initialized from scratch: {init_with_scratch}"
            )

    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    sd = torch.load(cfg.train.swinir_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in sd.items()
    }
    swinir.load_state_dict(sd, strict=True)
    for p in swinir.parameters():
        p.requires_grad = False
    if accelerator.is_main_process:
        print(f"load SwinIR from {cfg.train.swinir_path}")

    # set mode and cuda
    loaded_models['cldm'] = cldm.train().to(device)
    loaded_models['swinir'] = swinir.eval().to(device)
    
    # load text spotting module 
    if cfg.exp_args.model_name == 'terediff_stage2' or cfg.exp_args.model_name == 'terediff_stage3':
        sys.path.append(f'{os.getcwd()}/testr')
        from testr.adet.modeling.transformer_detector import TransformerDetector
        from testr.adet.config import get_cfg

        # get testr config
        config_testr = get_cfg()
        config_testr.merge_from_file(args.config_testr)
        config_testr.freeze()

        # load testr model
        detector = TransformerDetector(config_testr)

        # load testr pretrained weights
        if cfg.exp_args.testr_ckpt_dir is not None:
            ckpt = torch.load(cfg.exp_args.testr_ckpt_dir, map_location="cpu")
            load_result = detector.load_state_dict(ckpt['model'], strict=False)
            
            if accelerator.is_main_process:
                print("Loaded TESTR checkpoint keys:")
                print(" - Missing keys:", load_result.missing_keys)

        loaded_models['testr'] = detector.train().to(device)
    
    
    # load VLM for inference 
    if cfg.prompter_args.use_vlm_prompt:
        
        if cfg.vlm_args.vlm_name == 'qwenVL3B':
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
            vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map='auto')
            vlm_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        
        elif cfg.vlm_args.vlm_name == 'qwenVL7B':
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            from qwen_vl_utils import process_vision_info
            
            vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map='auto')
            vlm_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
        loaded_models['vlm_model'] = vlm_model.eval()
        loaded_models['vlm_processor'] = vlm_processor
    

    # -------------------------------- RESUME TRAINING ---------------------------------------
    if cfg.exp_args['resume_ckpt_dir'] is not None:
        ckpt_dir = f"{cfg.exp_args['resume_ckpt_dir']}"        
        ckpt=torch.load(ckpt_dir, map_location="cpu")
        for model_name, model in loaded_models.items():
            if model_name in ckpt:
                missing, unexpected = model.load_state_dict(ckpt[model_name], strict=False)
                print(f"RESUME TRAINING - Loaded {model_name} | Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
            else:
                print(f"Warning: No checkpoint found for {model_name}")
        for name, model in loaded_models.items():
            if isinstance(model, nn.Module):
                # Skip .to(device) for models that are offloaded via device_map
                if name.startswith("vlm_"):
                    continue
                model.to(device)
        return loaded_models, ckpt_dir

    return loaded_models, None


def set_training_params(accelerator, models, cfg):

    train_params=[]
    all_model_names=[]
    train_model_names=[]

    for model_name, model in models.items():

        for name, param in model.named_parameters():
            all_model_names.append(name)
            
            # stage1 training (training the ctrlnet and unet attention layers of the image restoration module)
            if cfg.exp_args.finetuning_method == 'image_restoration_module':
                if 'controlnet' in name or ('unet' in name and 'attn' in name):
                    param.requires_grad = True
                    train_model_names.append(name)
                    train_params.append(param)
                else:
                    param.requires_grad = False
            
            # stage2 training (training the text spotting module)
            elif cfg.exp_args.finetuning_method == 'text_spotting_module':
                if 'testr' in name:
                    param.requires_grad = True
                    train_model_names.append(name)
                    train_params.append(param)
                else:
                    param.requires_grad = False
            
            # stage3 training (training both the image restoration and text spotting modules)
            elif cfg.exp_args.finetuning_method == 'all_modules':
                if 'controlnet' in name or ('unet' in name and 'attn' in name) or ('testr' in name):
                    param.requires_grad = True
                    train_model_names.append(name)
                    train_params.append(param)
                else:
                    param.requires_grad = False

    # print modules to be trained
    if accelerator.is_main_process:
        print('================================================================= MODELS TO BE TRAINED =================================================================')
        chunk_size = 10  # Adjust based on readability
        for i in range(0, len(train_model_names), chunk_size):
            print(train_model_names[i:i+chunk_size])  # Print in smaller chunks
    
    return train_params, train_model_names



