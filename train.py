import cv2 
import copy
import wandb
import pyiqa
import numpy as np
from tqdm import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from argparse import ArgumentParser
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import DistributedDataParallelKwargs

import torch
import torch.nn as nn

from terediff.sampler import SpacedSampler
from terediff.model import ControlLDM, Diffusion
from terediff.dataset.utils import encode, decode 
from terediff.utils.common import instantiate_from_config, to, log_txt_as_img
import initialize


def main(args):


    # set accelerator, seed, device, config
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(split_batches=False, kwargs_handlers=[kwargs])
    set_seed(25, device_specific=True)
    device = accelerator.device
    gen = torch.Generator(device)
    cfg = OmegaConf.load(args.config)


    # load logging tools and ckpt directory
    if accelerator.is_main_process:
        exp_dir, ckpt_dir, exp_name, writer = initialize.load_experiment_settings(accelerator, cfg)


    # load data
    train_ds, val_ds, train_loader, val_loader = initialize.load_data(accelerator, cfg)
    train_batch_transform = instantiate_from_config(cfg.train_batch_transform)
    val_batch_transform = instantiate_from_config(cfg.val_batch_transform)
    

    # load models
    models, resume_ckpt_path = initialize.load_model(accelerator, device, args, cfg)
    

    # set training params
    train_params, train_model_names = initialize.set_training_params(accelerator, models, cfg)


    # setup optimizer
    opt = torch.optim.AdamW(train_params, lr=cfg.train.learning_rate)


    # setup ddpm
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(device)
    sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)


    # setup accelerator    
    models = {k: accelerator.prepare(v) for k, v in models.items()}
    opt, train_loader, val_loader = accelerator.prepare(opt, train_loader, val_loader)


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


    # print Training Info
    if accelerator.is_main_process:
        print('='*100)
        print(f'Experiment name: {exp_name}')
        print('-'*50)
        print(f"Save ckpt directory: {exp_dir}")
        print(f"Training steps: {cfg.train.train_steps}")
        print('-'*50)
        print(f"Num train_dataset: {len(train_ds):,}")
        print(f"Num val_dataset: {len(val_ds):,}")
        print('-'*50)
        print(f'Loaded models: {list(models.keys())}')
        print(f'Finetuning Method: {cfg.exp_args.finetuning_method}')
        print('-'*50)
        print(f'Resume training ckpt: ', resume_ckpt_path)
        print(f'OCR pretrained ckpt: {cfg.exp_args.testr_ckpt_dir}')
        print('-'*50)
        print(f'OCR loss weight: ', cfg.exp_args.ocr_loss_weight)
        print('='*100)


    # setup variables for monitoring/logging purposes:
    max_steps = cfg.train.train_steps
    global_step = 0
    epoch = 0
    
    diffusion_loss = 0.0
    ocr_loss = 0.0 
    ocr_losses={}         
    logging_counter=0
    
    total_step_loss = 0.0
    total_epoch_loss = 0.0
    

    # Training Loop
    while global_step < max_steps:

        # TRAINING
        pbar = tqdm( iterable=None, disable=not accelerator.is_main_process, unit="batch", total=len(train_loader), )
        for batch in train_loader:

            # log basic info while training
            if accelerator.is_main_process and cfg.log_args.log_tool == 'wandb':
                wandb.log({'global_step': global_step, 'epoch': epoch,'learning_rate': opt.param_groups[0]['lr'], })

            # load training data
            to(batch, device)
            batch = train_batch_transform(batch)
            gt, lq, train_prompt, texts, boxes, polys, text_encs, img_name = batch
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float()   # b 3 512 512 [-1,1]
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()   # b 3 512 512 [0,1]
            train_bs = gt.shape[0]
            
            # prepare VAE, condition, timestep
            with torch.no_grad():
                z_0 = pure_cldm.vae_encode(gt)                              # b 4 64 64
                clean = models['swinir'](lq)                                # b 3 512 512
                cond = pure_cldm.prepare_condition(clean, train_prompt)     # cond['c_txt'], cond['c_img']
                # noise augmentation
                cond_aug = copy.deepcopy(cond)

            # sample random training timesteps and obtain diffusion loss
            t = torch.randint(0, diffusion.num_timesteps, (train_bs,), device=device)
            diff_loss, extracted_feats = diffusion.p_losses(models['cldm'], z_0, t, cond_aug, cfg)
            
            breakpoint()
            if cfg.exp_args.model_name == 'terediff_stage2' or cfg.exp_args.model_name == 'terediff_stage3':
                # process annotations for OCR training loss
                train_targets=[]
                for i in range(train_bs):
                    num_box=len(boxes[i])
                    tmp_dict={}
                    tmp_dict['labels'] = torch.tensor([0]*num_box).cuda()  # 0 for text
                    tmp_dict['boxes'] = torch.tensor(boxes[i]).cuda()
                    tmp_dict['texts'] = text_encs[i]
                    tmp_dict['ctrl_points'] = polys[i]
                    train_targets.append(tmp_dict)
                # OCR model forward pass
                ocr_loss_dict, _ = models['testr'](extracted_feats, train_targets)
                # OCR total_loss
                ocr_tot_loss = sum(ocr_loss_dict.values())
                # OCR losses
                for ocr_key, ocr_val in ocr_loss_dict.items():
                    if ocr_key in ocr_losses.keys():
                        ocr_losses[ocr_key].append(ocr_val.item())
                    else:
                        ocr_losses[ocr_key]=[ocr_val.item()]
                total_loss = diff_loss + cfg.exp_args.ocr_loss_weight * ocr_tot_loss      
            else:
                total_loss = diff_loss
                ocr_tot_loss=torch.tensor(0).cuda()


            # calculate gradient and update model
            opt.zero_grad()
            accelerator.backward(total_loss)
            opt.step()
            accelerator.wait_for_everyone()
            global_step += 1


            # gather losses for logging
            diffusion_loss += diff_loss.item() 
            ocr_loss += ocr_tot_loss.item()
            total_step_loss += total_loss.item() 
            total_epoch_loss += total_loss.item()
            logging_counter += 1    


            # set terminal logging visualization
            pbar.update(1)
            pbar.set_description(f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, Diff_Loss: {diff_loss.item():.6f}")

            # log training loss
            if global_step % cfg.train.log_loss_every == 0 and global_step > 0:
                # Always do reduce on all processes
                avg_diffusion_loss = accelerator.reduce(torch.tensor(diffusion_loss / logging_counter, device=device), reduction="mean").item()
                avg_ocr_loss = accelerator.reduce(torch.tensor(ocr_loss / logging_counter, device=device), reduction="mean").item()
                avg_total_step_loss = accelerator.reduce(torch.tensor(total_step_loss / logging_counter, device=device), reduction="mean").item()

                # Log OCR components: reduce on all ranks, wandb.log on main process
                for ocr_key, ocr_val in ocr_losses.items():
                    if len(ocr_val) > 0:
                        avg_val = sum(ocr_val) / len(ocr_val)
                        avg_val_tensor = accelerator.reduce(
                            torch.tensor(avg_val, device=device), reduction="mean"
                        )
                        if accelerator.is_main_process and cfg.log_args.log_tool == 'wandb':
                            wandb.log({ f"train_loss_ocr_components/{ocr_key}": avg_val_tensor.item() })
                    ocr_val.clear()

                # Log summary losses
                if accelerator.is_main_process and cfg.log_args.log_tool == 'wandb':
                    wandb.log({"train_loss/diffusion_loss": avg_diffusion_loss})
                    wandb.log({"train_loss/ocr_tot_loss": avg_ocr_loss})
                    wandb.log({"train_loss/total_step_loss": avg_total_step_loss})

                # Reset counters
                diffusion_loss = 0.0
                ocr_loss = 0.0
                total_step_loss = 0.0
                logging_counter = 0


            # ======================== SAVE MODEL ========================
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_main_process:
                    ckpt = {}

                    # Unwrap models before saving their state_dicts
                    for model_name, model in models.items():
                        unwrapped_model = accelerator.unwrap_model(model)
                        ckpt[model_name] = unwrapped_model.state_dict()

                    ckpt_path = f"{ckpt_dir}/{global_step:07d}.pt"
                    torch.save(ckpt, ckpt_path)
            # =============================================================


            # log validation images 
            # if global_step % cfg.val.log_image_every == 0 or global_step == 1:
            if global_step % cfg.val.log_image_every == 0:

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

                # Validation
                for val_batch_idx, val_batch in enumerate(val_loader):

                    # load val data
                    to(val_batch, device)
                    val_batch = val_batch_transform(val_batch)
                    val_gt, val_lq, val_prompt, val_texts, val_boxes, val_polys, val_text_encs, val_img_name = val_batch 
                    val_gt = rearrange(val_gt, "b h w c -> b c h w").contiguous().float()   # b 3 512 512
                    val_lq = rearrange(val_lq, "b h w c -> b c h w").contiguous().float()   # b 3 512 512 
                    
                    val_bs, _, val_H, val_W = val_gt.shape
                    
                    # put models on evaluation for sampling
                    for model in models.values():
                        if isinstance(model, nn.Module):
                            model.eval()

                    # prepare vae, condition
                    with torch.no_grad():
                        # val_z_0 = pure_cldm.vae_encode(val_gt)
                        val_clean = models['swinir'](val_lq)    # b 3 512 512
                        val_cond = pure_cldm.prepare_condition(val_clean, val_prompt)

                        # set number of val imgs to log
                        M = cfg.val.log_num_val_img
                        val_log_clean = val_clean[:M]
                        val_log_cond = {k: v[:M] for k, v in val_cond.items()}
                        val_log_gt, val_log_lq = val_gt[:M], val_lq[:M]
                        val_log_prompt = val_prompt[:M]

                        pure_noise = torch.randn((M, 4, 64, 64), generator=gen, device=device, dtype=torch.float32)
                        
                        # sampling
                        val_z, val_sampled_unet_feats = sampler.sample(     # b 4 56 56
                            model=models['cldm'],
                            device=device,
                            steps=50,
                            x_size=(val_bs, 4, int(val_H/8), int(val_W/8)),   # manual shape adjustment
                            cond=val_log_cond,
                            uncond=None,
                            cfg_scale=1.0,
                            x_T = pure_noise,
                            progress=accelerator.is_main_process,
                            cfg=cfg
                        )

                        # =========================== OCR ===========================
                        if cfg.exp_args.model_name == 'terediff_stage2' or cfg.exp_args.model_name == 'terediff_stage3':

                            # process annotations for OCR val loss 
                            val_targets=[]
                            for i in range(val_bs):
                                num_box=len(val_boxes[i])
                                tmp_dict={}
                                tmp_dict['labels'] = torch.tensor([0]*num_box).cuda()  # 0 for text
                                tmp_dict['boxes'] = torch.tensor(val_boxes[i]).cuda()
                                tmp_dict['texts'] = val_text_encs[i]
                                tmp_dict['ctrl_points'] = val_polys[i]
                                val_targets.append(tmp_dict)


                            # evaluate diffusion features for different timesteps
                            for sampled_iter, sampled_timestep, unet_feats in val_sampled_unet_feats:

                                # OCR model forward pass
                                sampling_val_ocr_loss_dict, sampling_val_ocr_results = models['testr'](unet_feats, val_targets)
                                # val ocr total loss
                                sampling_val_ocr_tot_loss = sum(sampling_val_ocr_loss_dict.values())


                                # log sampling train loss and box to wandb
                                if accelerator.is_main_process and cfg.log_args.log_tool == 'wandb':
                                    for ocr_key, ocr_val in sampling_val_ocr_loss_dict.items():
                                        wandb.log({f"sampling_val_LOSS_iter{sampled_iter}_timestep{sampled_timestep}/{ocr_key}": ocr_val.item()})
                                    wandb.log({f"sampling_val_LOSS_iter{sampled_iter}_timestep{sampled_timestep}/ocr_tot_loss": sampling_val_ocr_tot_loss.item()})


                                # vis poly and text
                                for i in range(M):
                                    vis_val_gt = val_gt[i]                                  # 3 512 512 [-1,1]
                                    vis_val_gt = (vis_val_gt + 1)/2 * 255.0                 # 3 512 512 [0,255]
                                    vis_val_gt = vis_val_gt.permute(1,2,0).detach().cpu().numpy().astype(np.uint8).copy()  # 512 512 3

                                    results_per_img = sampling_val_ocr_results[i]

                                    for j in range(len(results_per_img.polygons)):
                                        val_ctrl_pnt= results_per_img.polygons[j].view(16,2).cpu().detach().numpy().astype(np.int32)    # 32 -> 16 2
                                        val_score = results_per_img.scores[j]                    
                                        val_rec = results_per_img.recs[j]
                                        val_pred_text = decode(val_rec)

                                        cv2.polylines(vis_val_gt, [val_ctrl_pnt], True, (0,255,0), 2)
                                        cv2.putText(vis_val_gt, val_pred_text, (val_ctrl_pnt[0][0], val_ctrl_pnt[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                                    # cv2.imwrite(f'./tmp{i}.jpg', vis_val_gt[...,::-1])
                                    if accelerator.is_main_process and cfg.log_args.log_tool == 'wandb':
                                        wandb.log({f'sampling_val_VIS_iter{sampled_iter}_timestep{sampled_timestep}/{val_batch_idx}_poly{i}': wandb.Image(vis_val_gt, caption=f'draw sampled val ocr results on gt')})
                                        

                        # log total psnr, ssim, lpips for val
                        tot_val_psnr.append(torch.mean(metric_psnr(torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item())
                        tot_val_ssim.append(torch.mean(metric_ssim(torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item())
                        tot_val_lpips.append(torch.mean(metric_lpips(torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item())
                        tot_val_dists.append(torch.mean(metric_dists(torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item())
                        tot_val_niqe.append(torch.mean(metric_niqe(torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item())
                        tot_val_musiq.append(torch.mean(metric_musiq(torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item())
                        tot_val_maniqa.append(torch.mean(metric_maniqa(torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item())
                        tot_val_clipiqa.append(torch.mean(metric_clipiqa(torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item())
                        

                        # log sampling val imgs to wandb
                        if accelerator.is_main_process and cfg.log_args.log_tool == 'wandb':

                            # log sampling val metrics 
                            wandb.log({f'sampling_val_METRIC/val_psnr': torch.mean(metric_psnr(
                                                                                            torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),
                                                                                            torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item(),
                                    f'sampling_val_METRIC/val_ssim': torch.mean(metric_ssim(
                                                                                            torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),
                                                                                            torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item(),
                                    f'sampling_val_METRIC/val_lpips': torch.mean(metric_lpips(
                                                                                            torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),
                                                                                            torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item(),
                                    f'sampling_val_METRIC/val_dists': torch.mean(metric_dists(
                                                                                            torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),
                                                                                            torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item(),
                                    f'sampling_val_METRIC/val_niqe': torch.mean(metric_niqe(
                                                                                            torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),
                                                                                            torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item(),
                                    f'sampling_val_METRIC/val_musiq': torch.mean(metric_musiq(
                                                                                            torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),
                                                                                            torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item(),
                                    f'sampling_val_METRIC/val_maniqa': torch.mean(metric_maniqa(
                                                                                            torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),
                                                                                            torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item(),
                                    f'sampling_val_METRIC/val_clipiqa': torch.mean(metric_clipiqa(
                                                                                            torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1),
                                                                                            torch.clamp((val_log_gt + 1) / 2, min=0, max=1))).item(),
                                    })
                            
                            # log sampling val images 
                            wandb.log({ f'sampling_val_FINAL_VIS/{val_batch_idx}_val_gt': wandb.Image((val_log_gt + 1) / 2, caption=f'gt_img'),
                                        f'sampling_val_FINAL_VIS/{val_batch_idx}_val_lq': wandb.Image(val_log_lq, caption=f'lq_img'),
                                        f'sampling_val_FINAL_VIS/{val_batch_idx}_val_cleaned': wandb.Image(val_log_clean, caption=f'cleaned_img'),
                                        f'sampling_val_FINAL_VIS/{val_batch_idx}_val_sampled': wandb.Image(torch.clip((pure_cldm.vae_decode(val_z) + 1) / 2, 0, 1), caption=f'sampled_img'),
                                        f'sampling_val_FINAL_VIS/{val_batch_idx}_val_prompt': wandb.Image(log_txt_as_img((256, 256), val_log_prompt), caption=f'prompt'),
                                    })
                            wandb.log({f'sampling_val_FINAL_VIS/{val_batch_idx}_val_all': wandb.Image(torch.concat([val_log_lq, val_log_clean, torch.clip((pure_cldm.vae_decode(val_z) + 1) / 2, 0, 1), val_log_gt], dim=2), caption='lq_clean_sample,gt')})

                    # put models back to training 
                    for model in models.values():
                        if isinstance(model, nn.Module):
                            model.train()
                    
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
                

            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break


        pbar.close()
        epoch += 1
        avg_total_epoch_loss = accelerator.reduce(torch.tensor(total_epoch_loss / len(train_loader), device=device), reduction="mean").item()
        total_epoch_loss = 0.0
        if accelerator.is_main_process and cfg.log_args.log_tool == 'wandb':
            wandb.log({"train_loss/total_epoch_loss": avg_total_epoch_loss})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument('--config_testr', type=str)
    args = parser.parse_args()
    main(args)
