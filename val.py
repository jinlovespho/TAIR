import re
import os
import csv
import json
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
from terediff.dataset.utils import encode, decode 
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
        cfg.exp_name = exp_name
    
    
    # load data annotation
    if cfg.dataset.val_dataset_name == 'RealText' or \
        cfg.dataset.val_dataset_name == 'SATextLv1' or \
            cfg.dataset.val_dataset_name == 'SATextLv2' or \
                cfg.dataset.val_dataset_name == 'SATextLv3':

        gt_imgs = sorted(os.listdir(f'{cfg.dataset.gt_img_path}'))  
        lq_imgs = sorted(os.listdir(f'{cfg.dataset.lq_img_path}'))
        
        gt_imgs = sorted([img for img in gt_imgs if img.endswith('.jpg')])
        lq_imgs = sorted([img for img in lq_imgs if img.endswith('.jpg')])
        
        gt_imgs_path = sorted([f'{cfg.dataset.gt_img_path}/{img}' for img in gt_imgs])
        lq_imgs_path = sorted([f'{cfg.dataset.lq_img_path}/{img}' for img in lq_imgs])
        len_val_ds = len(gt_imgs)
        
        model_H = cfg.model_args.model_H
        model_W = cfg.model_args.model_W
        
        # load llava caption
        if cfg.prompter_args.use_llava_prompt:
            llava_anns = json.load(open(cfg.prompter_args.llava_prompt_dir, 'r'))
            # llava_dic={}
            # f = open(cfg.prompter_args.llava_prompt_dir, 'r')
            # llava = csv.reader(f)
            # llava = sorted(list(llava))

            # for lva in llava:
            #     lva_id=lva[0]
            #     lva_prompt=lva[1].split(',')[0]
            #     llava_dic[lva_id]=lva_prompt
        
        # load json 
        json_path = cfg.dataset.gt_ann_path 
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            json_data = sorted(json_data.items())
        
        val_gt_json = {}
        for idx, (img_id, img_anns) in enumerate(json_data):
            anns = img_anns['0']['text_instances']
                
            boxes=[]
            texts=[]
            text_encs=[]
            polys=[]
            prompts=[]
            
            for ann in anns:
                # process text 
                text = ann['text']
                count=0
                for char in text:
                    # only allow OCR english vocab: range(32,127)
                    if 32 <= ord(char) and ord(char) < 127:
                        count+=1
                        # print(char, ord(char))
                if count == len(text) and count < 26:
                    texts.append(text)
                    text_encs.append(encode(text))
                    assert text == decode(encode(text)), 'check text encoding !'
                else:
                    continue
                
                # process box
                box_xyxy = ann['bbox']
                x1,y1,x2,y2 = box_xyxy
                box_xywh = [ x1, y1, x2-x1, y2-y1 ]
                box_xyxy_scaled = list(map(lambda x: x/model_H, box_xyxy))  # scale box coord to [0,1]
                x1,y1,x2,y2 = box_xyxy_scaled 
                box_cxcywh = [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]   # xyxy -> cxcywh
                # # select box format
                # if cfg.dataset.data_args['bbox_format'] == 'xywh_unscaled':
                #     processed_box = box_xywh
                #     processed_box = list(map(lambda x: int(x), processed_box))
                # elif cfg.dataset.data_args['bbox_format'] == 'xyxy_scaled':
                #     processed_box = box_xyxy_scaled
                #     processed_box = list(map(lambda x: round(x,4), processed_box))
                # elif cfg.dataset.data_args['bbox_format'] == 'cxcywh_scaled':
                #     processed_box = box_cxcywh
                #     processed_box = list(map(lambda x: round(x,4), processed_box))
                processed_box = box_cxcywh
                processed_box = list(map(lambda x: round(x,4), processed_box))
                boxes.append(processed_box)
                
                # process polygon
                poly = np.array(ann['polygon']).astype(np.int32)    # 16 2
                # scale poly
                poly_scaled = poly / np.array([model_W, model_H])
                polys.append(poly_scaled)

            # check is anns are properly processed
            assert len(boxes) == len(texts) == len(text_encs) == len(polys), f" Check len"
            if len(boxes) == 0 or len(polys) == 0:
                    continue
            
            # process prompt
            caption = [f'"{txt}"' for txt in texts]
            # prompt = f"A high-quality photo containing the word {', '.join(caption) }."
            if cfg.prompter_args.prompt_style == 'CAPTION':
                prompt = f"A realistic scene where the texts {', '.join(caption) } appear clearly on signs, boards, buildings, or other objects."
            elif cfg.prompter_args.prompt_style == 'TAG':
                prompt = f"{', '.join(caption)}"
            
            # if cfg.prompter_args.use_llava_prompt:
            #     # prompt = llava_dic[img_id]
            #     prompt = llava_anns[img_id]['vlm_output']

            prompts.append(prompt)

            val_gt_json[img_id] = {
                'boxes': boxes,
                'texts': texts,
                'text_encs': text_encs,
                'polys': polys,
                'gtprompts': prompts
            }
    
    else:
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
    
    
    # set VLM 
    if cfg.prompter_args.use_vlm_prompt:
        vlm_model = models['vlm_model']
        vlm_processor = models['vlm_processor']
    else:
        vlm_model = None 
        vlm_processor = None 
    

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
    
    
    # print experiment info
    print("=" * 80)
    print(f"{'Experiment Config':^80}")
    print("=" * 80)
    print(f"{'Mode':25}: {cfg.exp_args.mode}")
    print(f"{'Model Name':25}: {cfg.exp_args.model_name}")
    print(f"{'Checkpoint Path':25}: {cfg.exp_args.resume_ckpt_dir}")
    print(f"{'Image Restoration Sample Steps':25}: {cfg.exp_args.inf_sample_step}")
    print("------------- Prompter Args -------------")
    for k, v in cfg.prompter_args.items():
        print(f"{k:25}: {v}")
    print("----------------- VLM Args --------------")
    for k, v in cfg.vlm_args.items():
        print(f"{k:25}: {v}")
    print("=" * 80)
    
    inf_time=[]
    inf_time_modules={'img_restoration_module':[], 'text_spotting_module':[]}
    
    # print model param info 
    tot_model_params=0
    for model_name, model in models.items():
        if isinstance(model, nn.Module):
            num_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"{model_name:25}: {num_train_params/1e6:.2f}M trainable parameters")
            
            tot_params = sum(p.numel() for p in model.parameters())
            tot_model_params += tot_params
            print(f"{model_name:25}: {tot_params/1e6:.2f}M total parameters")
    print(f"{'Total Model Params':25}: {tot_model_params/1e6:.2f}M")
    print("=" * 80)
    
    unet = sum(p.numel() for p in models['cldm'].unet.parameters()) / 1e6
    vae = sum(p.numel() for p in models['cldm'].vae.parameters())   / 1e6
    clip = sum(p.numel() for p in models['cldm'].clip.parameters()) / 1e6
    controlnet = sum(p.numel() for p in models['cldm'].controlnet.parameters()) / 1e6
    print(f"{'UNet Params':25}: {unet:.2f}M")
    print(f"{'VAE Params':25}: {vae:.2f}M")
    print(f"{'CLIP Params':25}: {clip:.2f}M")
    print(f"{'ControlNet Params':25}: {controlnet:.2f}M")
    print(f"{'Total CLDM Params':25}: {unet + vae + clip + controlnet:.2f}M")
    print("=" * 80)
    
    # # eval for 50 images 
    # num_eval_img=50
    # gt_imgs_path = gt_imgs_path[:num_eval_img] if len(gt_imgs_path) > num_eval_img else gt_imgs_path
    # lq_imgs_path = lq_imgs_path[:num_eval_img] if len(lq_imgs_path) > num_eval_img else lq_imgs_path
    # len_val_ds = len(gt_imgs_path)
    # print(f"Total validation images: {len_val_ds}")
    # print(f"Total validation images: {len(gt_imgs_path)}")
    

    for val_batch_idx, (gt_img_path, lq_img_path) in enumerate(tqdm(zip(gt_imgs_path, lq_imgs_path), desc='val', total=len(gt_imgs_path))):
        
        gt_id = gt_img_path.split('/')[-1].split('.')[0]
        lq_id = lq_img_path.split('/')[-1].split('.')[0]
        assert gt_id == lq_id, f"gt_img_path: {gt_img_path}, lq_img_path: {lq_img_path} do not match"
        
        gt_img = Image.open(gt_img_path)     # size: 512
        lq_img = Image.open(lq_img_path)     # size: 128
        
        val_gt = preprocess_gt(gt_img).unsqueeze(0).to(device)  # 1 3 512 512
        val_lq = preprocess_lq(lq_img).unsqueeze(0).to(device)  # 1 3 512 512
        val_bs, _, val_H, val_W = val_gt.shape
        
        
        # load gt annotation
        val_gt_box = val_gt_json[gt_id]['boxes']
        val_gt_text = val_gt_json[gt_id]['texts']
        val_gt_text_enc = val_gt_json[gt_id]['text_encs']
        val_gt_poly = val_gt_json[gt_id]['polys']
        val_gt_prompt = val_gt_json[gt_id]['gtprompts']
        
        
        # the inital prompt is null prompt
        val_prompt = [""]
        
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        
        # inference 
        with torch.no_grad():
            val_clean = models['swinir'](val_lq)   
            val_cond = pure_cldm.prepare_condition(val_clean, val_prompt)
            
            M=1
            pure_noise = torch.randn((1, 4, 64, 64), generator=gen, device=device, dtype=torch.float32)
            models['testr'].test_score_threshold = 0.5   
            ts_model = models['testr']
            
            # precomputed vlm caption anns(qwen, llava)
            if cfg.prompter_args.use_llava_prompt:
                print('Using precomputed LLAVA prompt')
                vlm_caption_dir = cfg.prompter_args.llava_prompt_dir 
                caption_anns = json.load(open(vlm_caption_dir, 'r'))
            elif cfg.prompter_args.use_qwen_prompt:
                print('Using precomputed QWENVL prompt')
                vlm_caption_dir = cfg.prompter_args.qwen_prompt_dir 
                caption_anns = json.load(open(vlm_caption_dir, 'r'))
            else:
                caption_anns={}

            # sampling
            val_z, val_ts_result, val_vlm_result = sampler.val_sample(    
                model=models['cldm'],
                device=device,
                steps=cfg.exp_args.inf_sample_step,
                x_size=(val_bs, 4, int(val_H/8), int(val_W/8)),
                cond=val_cond,
                uncond=None,
                cfg_scale=1.0,
                x_T = pure_noise,
                progress=accelerator.is_main_process,
                cfg=cfg, 
                pure_cldm=pure_cldm,
                ts_model = ts_model,
                val_gt_text = val_gt_text,
                vlm_model=vlm_model,
                vlm_processor=vlm_processor,
                lq_img = val_lq,
                cleaned_img = val_clean,
                inf_time_modules=inf_time_modules,
                vis_args = cfg.vis_args,
                val_gt=val_gt,
                img_id = lq_id,
                caption_anns=caption_anns
            )
            
            
            

            # log val prompts
            val_prompt = val_prompt[0]
            lines = []
            
            if cfg.prompter_args.use_gt_prompt:
                lines.append(f"** using GT prompt w/ {cfg.prompter_args.prompt_style}style **\n")
                lines.append(f'GT PROMPT: {val_gt_text}')
            elif cfg.prompter_args.use_ts_prompt:
                lines.append(f"** using TS prompt w/ {cfg.prompter_args.prompt_style}style **\n")
            elif cfg.prompter_args.use_edit_prompt:
                lines.append(f"** using Editting prompt w/ {cfg.prompter_args.prompt_style}style **\n")
            elif cfg.prompter_args.use_vlm_prompt:
                lines.append(f"** using VLM prompt w/ {cfg.prompter_args.prompt_style}style **\n")
                lines.append(f'VLM inference step: {cfg.vlm_args.inf_vlm_step}/{cfg.exp_args.inf_sample_step}')
                lines.append(f'VLM input image: {cfg.vlm_args.vlm_input_img}')
                lines.append(f'VLM input prompt: \n')
                width = 80
                for i in range(0, len(val_vlm_result['vlm_input_prompt']), width):
                    lines.append(val_vlm_result['vlm_input_prompt'][i:i+width] + "\n")
                lines.append("\n")
            elif cfg.prompter_args.use_null_prompt:
                lines.append(f"** using Null prompt w/ {cfg.prompter_args.prompt_style}style **\n")
                
            # Format prompt
            lines.append("initial input prompt:\n")
            width = 80
            for i in range(0, len(val_prompt), width):
                lines.append(val_prompt[i:i+width] + "\n")
            lines.append("\n")
            
            # Add prediction results
            for inf_step, ts_result in enumerate(val_ts_result):
                timestep = ts_result['timestep']
                pred_texts = ts_result['pred_texts']
                ts_pred_text = ts_result['ts_pred_text']
                
                if cfg.prompter_args.use_gt_prompt:
                    lines.append(f"timestep: {timestep:<4}  /  gt_text: {pred_texts}\n")
                elif cfg.prompter_args.use_ts_prompt:
                    lines.append(f"timestep: {timestep:<4}  /  ts_pred_text: {pred_texts}\n")
                elif cfg.prompter_args.use_edit_prompt:
                    lines.append(f"timestep: {timestep:<4}  /  edit_text: {pred_texts}\n")
                elif cfg.prompter_args.use_vlm_prompt:
                    if inf_step < cfg.vlm_args.inf_vlm_step:
                        lines.append(f"timestep: {timestep:<4}  /  vlm_pred_text: {pred_texts}  /  ts_pred_text: {ts_pred_text}\n")
                    else:
                        lines.append(f"timestep: {timestep:<4}  /  ts_pred_text: {pred_texts}\n")
                
            
            # save prompt as txt
            if cfg.exp_args.save_result_prompt:
                txt_save_path = f'{cfg.exp_args.save_val_img_dir}/txt'
                os.makedirs(txt_save_path, exist_ok=True)
                with open(f'{txt_save_path}/{gt_id}.txt', "w") as file:
                    for line in lines:
                        file.write(line)

            
            # Now convert the list of strings to image
            img_of_pred_text = text_to_image(lines)
            restored_img = torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1)   # 1 3 512 512
            
            end.record()
            torch.cuda.synchronize()
            inf_time.append(start.elapsed_time(end) / 1000)  # in seconds
            
            inf_time_avg = np.mean(inf_time)
            print(f'len inf_time: {len(inf_time)}')
            # print('inf time: ', inf_time)
            print('avg inf time: ', inf_time_avg)
            print('-'*30)
            
            img_module_avg = np.mean(inf_time_modules['img_restoration_module']) / 1000
            ts_module_avg = np.mean(inf_time_modules['text_spotting_module']) / 1000
            print(f'img_restoration_module avg time: {img_module_avg:.4f} seconds')
            print(f'text_spotting_module avg time: {ts_module_avg:.4f} seconds')
            print('-'*30)
            
            
            
            # save sampled image and pred text result 
            if cfg.log_args.log_tool is None:
                img_save_path = f'{cfg.exp_args.save_val_img_dir}/{exp_name}'
                os.makedirs(img_save_path, exist_ok=True)
                restored_img_pil = TF.to_pil_image(restored_img.squeeze().cpu())
                restored_img_pil.save(f'{img_save_path}/{gt_id}.png')
                img_of_pred_text.save(f'{img_save_path}/text_{gt_id}.png')
            
            
            # save sampled images only
            if cfg.exp_args.save_result_img:
                img_save_path = f'{cfg.exp_args.save_val_img_dir}/{exp_name}'
                os.makedirs(img_save_path, exist_ok=True)
                restored_img_pil = TF.to_pil_image(restored_img.squeeze().cpu())
                restored_img_pil.save(f'{img_save_path}/{gt_id}.png')

            
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

    print(f"Total validation images: {len_val_ds}")
    print(f"Total validation time: {np.sum(inf_time)} seconds")
    print(f"Average validation time: {inf_time_avg} seconds")
    print(f'Average image restoration module time: {img_module_avg} s')
    print(f'Average text spotting module time: {ts_module_avg} s')
    
        
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
