import os
import argparse
import wandb
import pyiqa
import numpy as np
from PIL import Image 
from tqdm import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from accelerate.utils import set_seed
from terediff.utils.common import instantiate_from_config, text_to_image
from terediff.model import ControlLDM, Diffusion
from terediff.sampler import SpacedSampler
import initialize


def main(args):


    # set accelerator, seed, device, config
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(split_batches=False, kwargs_handlers=[kwargs])
    set_seed(25, device_specific=False)
    device = accelerator.device
    gen = torch.Generator(device)
    cfg = OmegaConf.load(args.config)
    
    
    # load logging tools and ckpt directory
    if accelerator.is_main_process:
        _, _, exp_name, _ = initialize.load_experiment_settings(accelerator, cfg)
    
    
    # load demo images from demo_imgs/ folder
    gt_imgs_path = sorted([f"{cfg.dataset.gt_img_path}/{img}" for img in os.listdir(cfg.dataset.gt_img_path) if img.endswith(".jpg")])
    lq_imgs_path = sorted([f"{cfg.dataset.lq_img_path}/{img}" for img in os.listdir(cfg.dataset.lq_img_path) if img.endswith(".jpg")])

                    
    # load models
    models, _ = initialize.load_model(accelerator, device, args, cfg)
    

    # setup ddpm
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(device)
    sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)


    # setup model accelerator    
    models = {k: accelerator.prepare(v) for k, v in models.items()}


    # unwrap cldm from accelerator for proper model saving
    pure_cldm: ControlLDM = accelerator.unwrap_model(models['cldm'])


    # SR metrics
    metric_psnr = pyiqa.create_metric('psnr', device=device)
    metric_ssim = pyiqa.create_metric('ssimc', device=device)
    metric_lpips = pyiqa.create_metric('lpips', device=device)
    metric_dists = pyiqa.create_metric('dists', device=device)
    metric_niqe = pyiqa.create_metric('niqe', device=device)
    metric_musiq = pyiqa.create_metric('musiq', device=device)
    metric_maniqa = pyiqa.create_metric('maniqa', device=device)
    metric_clipiqa = pyiqa.create_metric('clipiqa', device=device)


    tot_val_psnr=[]
    tot_val_ssim=[]
    tot_val_lpips=[]
    tot_val_dists=[]
    tot_val_niqe=[]
    tot_val_musiq=[]
    tot_val_maniqa=[]
    tot_val_clipiqa=[]
    

    # set seed for identical generation for validation sampling noise
    gen.manual_seed(25)
    
    
    # put model on eval
    for model in models.values():
        if isinstance(model, nn.Module):
            model.eval()


    # For val_gt (range [-1, 1])
    preprocess_gt = T.Compose([
        T.Resize(size=(512, 512), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # For val_lq (range [0, 1])
    preprocess_lq = T.Compose([
        T.Resize(size=(512, 512), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor()
    ])
    
    for val_batch_idx, (gt_img_path, lq_img_path) in enumerate(tqdm(zip(gt_imgs_path, lq_imgs_path), desc='val', total=len(gt_imgs_path))):
        
        gt_id = gt_img_path.split('/')[-1].split('.')[0]
        lq_id = lq_img_path.split('/')[-1].split('.')[0]
        assert gt_id == lq_id, f"gt_img_path: {gt_img_path}, lq_img_path: {lq_img_path} do not match"
        
        gt_img = Image.open(gt_img_path)     # size: 512
        lq_img = Image.open(lq_img_path)     # size: 128
        
        val_gt = preprocess_gt(gt_img).unsqueeze(0).to(device)  # 1 3 512 512
        val_lq = preprocess_lq(lq_img).unsqueeze(0).to(device)  # 1 3 512 512
        val_bs, _, val_H, val_W = val_gt.shape
        
        val_prompt = [""]
        
        if cfg.exp_args.editting_text is not None:
            val_prompt = [cfg.exp_args.editting_text]
            
        with torch.no_grad():
            val_clean = models['swinir'](val_lq)   
            val_cond = pure_cldm.prepare_condition(val_clean, val_prompt)

            M=1
            pure_noise = torch.randn((1, 4, 64, 64), generator=gen, device=device, dtype=torch.float32)
            
            models['testr'].test_score_threshold = 0.5   
            ts_model = models['testr']
            
            # sampling
            val_z, val_ts_results = sampler.val_sample(    
                model=models['cldm'],
                device=device,
                steps=cfg.exp_args.inf_sample_step,
                x_size=(val_bs, 4, int(val_H/8), int(val_W/8)),   # manual shape adjustment
                cond=val_cond,
                uncond=None,
                cfg_scale=1.0,
                x_T = pure_noise,
                progress=accelerator.is_main_process,
                cfg=cfg, 
                pure_cldm=pure_cldm,
                ts_model = ts_model,
                val_prompt=val_prompt
            )
            
            # log val prompts
            val_prompt = val_prompt[0]
            lines = []
            lines.append(f"** using OCR prompt w/ {cfg.exp_args.prompt_style}style **\n")
            # Format prompt
            lines.append("initial input prompt:\n")
            width = 80
            for i in range(0, len(val_prompt), width):
                lines.append(val_prompt[i:i+width] + "\n")
            lines.append("\n")
            
            # Add prediction results
            for ts_result in val_ts_results:
                timestep = ts_result['timestep']
                if cfg.exp_args.editting_text is not None:
                    pred_texts = val_prompt 
                elif cfg.exp_args.editting_text is None:
                    pred_texts = ', '.join(ts_result['pred_texts'])
                lines.append(f"timestep: {timestep:<4} /  pred_texts: {pred_texts}\n")
            
            # Now convert the list of strings to image
            img_of_pred_text = text_to_image(lines)
            
            restored_img = torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1)   # 1 3 512 512
            
            # save sampled images   
            if cfg.log_args.log_tool is None:
                img_save_path = f'{cfg.exp_args.save_val_img_dir}'
                os.makedirs(img_save_path, exist_ok=True)
                restored_img_pil = TF.to_pil_image(restored_img.squeeze().cpu())
                restored_img_pil.save(f'{img_save_path}/restored_{gt_id}.png')
                img_of_pred_text.save(f'{img_save_path}/pred_texts_{gt_id}.png')
            
            
            # log total psnr, ssim, lpips for val
            tot_val_psnr.append(torch.mean(metric_psnr(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
            tot_val_ssim.append(torch.mean(metric_ssim(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
            tot_val_lpips.append(torch.mean(metric_lpips(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
            tot_val_dists.append(torch.mean(metric_dists(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
            tot_val_niqe.append(torch.mean(metric_niqe(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
            tot_val_musiq.append(torch.mean(metric_musiq(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
            tot_val_maniqa.append(torch.mean(metric_maniqa(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
            tot_val_clipiqa.append(torch.mean(metric_clipiqa(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
            
            # log sampling val imgs to wandb
            if accelerator.is_main_process and cfg.log_args.log_tool == 'wandb':

                # log sampling val metrics 
                wandb.log({f'sampling_val_METRIC/val_psnr': torch.mean(metric_psnr(
                                                                                restored_img, 
                                                                                torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                        f'sampling_val_METRIC/val_ssim': torch.mean(metric_ssim(
                                                                                restored_img, 
                                                                                torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                        f'sampling_val_METRIC/val_lpips': torch.mean(metric_lpips(
                                                                                restored_img, 
                                                                                torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                        f'sampling_val_METRIC/val_dists': torch.mean(metric_dists(
                                                                                restored_img, 
                                                                                torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                        f'sampling_val_METRIC/val_niqe': torch.mean(metric_niqe(
                                                                                restored_img, 
                                                                                torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                        f'sampling_val_METRIC/val_musiq': torch.mean(metric_musiq(
                                                                                restored_img, 
                                                                                torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                        f'sampling_val_METRIC/val_maniqa': torch.mean(metric_maniqa(
                                                                                restored_img, 
                                                                                torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                        f'sampling_val_METRIC/val_clipiqa': torch.mean(metric_clipiqa(
                                                                                restored_img, 
                                                                                torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                        })
                
                # log sampling val images 
                wandb.log({ f'sampling_val_FINAL_VIS/{gt_id}_val_gt': wandb.Image((val_gt + 1) / 2, caption=f'gt_img'),
                            f'sampling_val_FINAL_VIS/{gt_id}_val_lq': wandb.Image(val_lq, caption=f'lq_img'),
                            f'sampling_val_FINAL_VIS/{gt_id}_val_cleaned': wandb.Image(val_clean, caption=f'cleaned_img'),
                            f'sampling_val_FINAL_VIS/{gt_id}_val_sampled': wandb.Image(torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, 0, 1), caption=f'sampled_img'),
                            f'sampling_val_FINAL_VIS/{gt_id}_val_prompts': wandb.Image(img_of_pred_text, caption='prompts used for sampling'),
                        })
                wandb.log({f'sampling_val_FINAL_VIS/{gt_id}_val_all': wandb.Image(torch.concat([val_lq, val_clean, torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, 0, 1), val_gt], dim=2), caption='lq_clean_sample,gt')})

        
    # average using numpy
    tot_val_psnr = np.array(tot_val_psnr).mean()
    tot_val_ssim = np.array(tot_val_ssim).mean()
    tot_val_lpips = np.array(tot_val_lpips).mean()
    tot_val_dists = np.array(tot_val_dists).mean()
    tot_val_niqe = np.array(tot_val_niqe).mean()
    tot_val_musiq = np.array(tot_val_musiq).mean()
    tot_val_maniqa = np.array(tot_val_maniqa).mean()
    tot_val_clipiqa = np.array(tot_val_clipiqa).mean()


    # log total val metrics 
    if accelerator.is_main_process and cfg.log_args.log_tool == 'wandb':
        wandb.log({
            f'sampling_val_METRIC/tot_val_psnr': tot_val_psnr,
            f'sampling_val_METRIC/tot_val_ssim': tot_val_ssim,
            f'sampling_val_METRIC/tot_val_lpips': tot_val_lpips,
            f'sampling_val_METRIC/tot_val_dists': tot_val_dists,
            f'sampling_val_METRIC/tot_val_niqe': tot_val_niqe,
            f'sampling_val_METRIC/tot_val_musiq': tot_val_musiq,
            f'sampling_val_METRIC/tot_val_maniqa': tot_val_maniqa,
            f'sampling_val_METRIC/tot_val_clipiqa': tot_val_clipiqa,
        })
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument('--config_testr', type=str)
    args = parser.parse_args()
    main(args)
