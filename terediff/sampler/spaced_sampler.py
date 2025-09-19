import re 
import cv2 

from typing import Optional, Tuple, Dict, Literal

import torch
import numpy as np
from tqdm import tqdm

from .sampler import Sampler
from ..model.gaussian_diffusion import extract_into_tensor
from ..model.cldm import ControlLDM
from terediff.dataset.utils import encode, decode 

from torchvision.utils import save_image 
from qwen_vl_utils import process_vision_info
import torchvision.transforms.functional as TF
from PIL import Image
import os

from terediff.model.open_clip import tokenize
from terediff.model.open_clip.tokenizer import SimpleTokenizer

import torch.nn.functional as F 


# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/respace.py
def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedSampler(Sampler):

    def __init__(
        self,
        betas: np.ndarray,
        parameterization: Literal["eps", "v"],
        rescale_cfg: bool,
    ) -> "SpacedSampler":
        super().__init__(betas, parameterization, rescale_cfg)

    def make_schedule(self, num_steps: int) -> None:
        used_timesteps = space_timesteps(self.num_timesteps, str(num_steps))
        betas = []
        last_alpha_cumprod = 1.0
        for i, alpha_cumprod in enumerate(self.training_alphas_cumprod):
            if i in used_timesteps:
                betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
        self.timesteps = np.array(
            sorted(list(used_timesteps)), dtype=np.int32
        )  # e.g. [0, 10, 20, ...]

        betas = np.array(betas, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if len(posterior_variance) > 1:
            posterior_log_variance_clipped = np.log(
                np.append(posterior_variance[1], posterior_variance[1:])
            )
        else:
            posterior_log_variance_clipped = np.log(
                np.append(posterior_variance[0], posterior_variance[0])
            )
        posterior_mean_coef1 = (
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        self.register("sqrt_alphas_cumprod", np.sqrt(alphas_cumprod))
        self.register("sqrt_one_minus_alphas_cumprod", np.sqrt(1 - alphas_cumprod))
        self.register("sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod)
        self.register("sqrt_recipm1_alphas_cumprod", sqrt_recipm1_alphas_cumprod)
        self.register("posterior_variance", posterior_variance)
        self.register("posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register("posterior_mean_coef1", posterior_mean_coef1)
        self.register("posterior_mean_coef2", posterior_mean_coef2)

    def q_posterior_mean_variance(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        return mean, variance

    def _predict_xstart_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_v(
        self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def apply_model(
        self,
        model: ControlLDM,
        x: torch.Tensor,
        model_t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        uncond: Optional[Dict[str, torch.Tensor]],
        cfg_scale: float,
    ) -> torch.Tensor:
        if uncond is None or cfg_scale == 1.0:
            model_output = model(x, model_t, cond)
        else:
            model_cond = model(x, model_t, cond)
            model_uncond = model(x, model_t, uncond)
            model_output = model_uncond + cfg_scale * (model_cond - model_uncond)
        return model_output

    @torch.no_grad()
    def p_sample(
        self,
        model: ControlLDM,
        x: torch.Tensor,
        model_t: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        uncond: Optional[Dict[str, torch.Tensor]],
        cfg_scale: float,
    ) -> torch.Tensor:
        # predict x_0
        model_output, extracted_feats = self.apply_model(model, x, model_t, cond, uncond, cfg_scale)
        if self.parameterization == "eps":
            pred_x0 = self._predict_xstart_from_eps(x, t, model_output)
        else:
            pred_x0 = self._predict_xstart_from_v(x, t, model_output)   # b 4 64 64 
        # calculate mean and variance of next state
        mean, variance = self.q_posterior_mean_variance(pred_x0, x, t)
        # sample next state
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        x_prev = mean + nonzero_mask * torch.sqrt(variance) * noise
        return x_prev, extracted_feats

    @torch.no_grad()
    def val_p_sample(
        self,
        model: ControlLDM,
        x: torch.Tensor,
        model_t: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        uncond: Optional[Dict[str, torch.Tensor]],
        cfg_scale: float,
    ) -> torch.Tensor:
        # predict x_0
        model_output, extracted_feats = self.apply_model(model, x, model_t, cond, uncond, cfg_scale)
        if self.parameterization == "eps":
            pred_x0 = self._predict_xstart_from_eps(x, t, model_output)
        else:
            pred_x0 = self._predict_xstart_from_v(x, t, model_output)   # b 4 64 64 
        # calculate mean and variance of next state
        mean, variance = self.q_posterior_mean_variance(pred_x0, x, t)
        # sample next state
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        x_prev = mean + nonzero_mask * torch.sqrt(variance) * noise
        return x_prev, extracted_feats, pred_x0

    @torch.no_grad()
    def sample(
        self,
        model: ControlLDM,
        device: str,
        steps: int,
        x_size: Tuple[int],
        cond: Dict[str, torch.Tensor],
        uncond: Dict[str, torch.Tensor],
        cfg_scale: float,
        tiled: bool = False,
        tile_size: int = -1,
        tile_stride: int = -1,
        x_T: torch.Tensor | None = None,
        progress: bool = True,
        cfg=None,
    ) -> torch.Tensor:

        self.make_schedule(steps)
        self.to(device)

        if x_T is None: # t
            x_T = torch.randn(x_size, device=device, dtype=torch.float32)

        x = x_T
        timesteps = np.flip(self.timesteps)
        total_steps = len(self.timesteps)
        iterator = tqdm(timesteps, total=total_steps, disable=not progress)
        bs = x_size[0]

        sampling_steps = cfg.exp_args['unet_feat_sampling_timestep']
        sampled_unet_feats = []

        for i, current_timestep in enumerate(iterator):
            # print(i, timestep)
            model_t = torch.full((bs,), current_timestep, device=device, dtype=torch.long)
            t = torch.full((bs,), total_steps - i - 1, device=device, dtype=torch.long)
            cur_cfg_scale = self.get_cfg_scale(cfg_scale, current_timestep)
            x, extracted_feats = self.p_sample(
                model,
                x,
                model_t,
                t,
                cond,
                uncond,
                cur_cfg_scale,
            )

            # JLP 
            if i+1 in sampling_steps:
                sampled_unet_feats.append( (i+1, current_timestep, extracted_feats) )

        return x, sampled_unet_feats 

    @torch.no_grad()
    def val_sample(
        self,
        model: ControlLDM,
        device: str,
        steps: int,
        x_size: Tuple[int],
        cond: Dict[str, torch.Tensor],
        uncond: Dict[str, torch.Tensor],
        cfg_scale: float,
        tiled: bool = False,
        tile_size: int = -1,
        tile_stride: int = -1,
        x_T: torch.Tensor | None = None,
        progress: bool = True,
        cfg=None,
        pure_cldm=None,
        ts_model=None,
        val_gt_text=None,
        vlm_model=None,
        vlm_processor=None,
        lq_img=None,
        cleaned_img=None,
        inf_time_modules=None,
        vis_args=None,
        val_gt=None,
        img_id=None,
        **kwargs
    ) -> torch.Tensor:

        self.make_schedule(steps)
        self.to(device)
        
        if x_T is None: # t
            x_T = torch.randn(x_size, device=device, dtype=torch.float32)

        x = x_T
        timesteps = np.flip(self.timesteps)
        total_steps = len(self.timesteps)
        iterator = tqdm(timesteps, total=total_steps, disable=not progress)
        bs = x_size[0]
        assert ts_model is not None, "Text-spotting model must be provided for validation sampling."
        
        
        img_inf_time = []
        ts_inf_time = []
        
        
        ts_results=[]
        vlm_results={}
        for i, current_timestep in enumerate(iterator):
            model_t = torch.full((bs,), current_timestep, device=device, dtype=torch.long)
            t = torch.full((bs,), total_steps - i - 1, device=device, dtype=torch.long)
            cur_cfg_scale = self.get_cfg_scale(cfg_scale, current_timestep)
            
            start1 = torch.cuda.Event(enable_timing=True)
            end1 = torch.cuda.Event(enable_timing=True)
            start1.record() 
            x, extracted_feats, pred_z0 = self.val_p_sample(
                model,
                x,
                model_t,
                t,
                cond,
                uncond,
                cur_cfg_scale,
            )
            end1.record()
            torch.cuda.synchronize()
            img_inf_time.append(start1.elapsed_time(end1))  # ms
            

            # visualize ca map between img and text
            if vis_args.vis_attn_map and i>0 and current_timestep in vis_args.vis_diff_timesteps:

                # make save dir 
                save_dir1 = f'{vis_args.attn_map_save_dir}/ctrlnet/timestep{current_timestep}'
                save_dir2 = f'{vis_args.attn_map_save_dir}/unet/timestep{current_timestep}'
                save_dir3 = f'{vis_args.attn_map_save_dir}/ctrlnet_unet/timestep{current_timestep}'
                os.makedirs(save_dir1, exist_ok=True)
                os.makedirs(save_dir2, exist_ok=True)
                os.makedirs(save_dir3, exist_ok=True)

                tmp_tokenizer = SimpleTokenizer()
                gt_img = np.array(val_gt.squeeze().permute(1,2,0).detach().cpu())

                if len(pred_txt) > 0:
                    # collect ctrlnet camaps
                    ctrlnet_camaps=[]
                    ctrlnet = model.controlnet
                    for name, module in ctrlnet.named_modules():
                        if name.endswith('attn2'):
                            cleaned_name = re.sub(r'\.(\d+)', r'[\1]', name)
                            attn_module = eval(f'ctrlnet.{cleaned_name}')
                            ctrlnet_camaps.append(attn_module.ca_map)
                    # collect unet camaps
                    unet_camaps=[]
                    unet = model.unet
                    for name, module in unet.named_modules():
                        if name.endswith('attn2'):
                            cleaned_name = re.sub(r'\.(\d+)', r'[\1]', name)
                            attn_module = eval(f'unet.{cleaned_name}')
                            unet_camaps.append(attn_module.ca_map)
                    # tokenize texts
                    prompt_tkn = tokenize(pred_prompt)  # 1 77  
                    ids = prompt_tkn[0]
                    ids = ids[ids != 0].tolist()

                    # save attention maps 
                    if vis_args.avg_attn_layers:
                        # ------------------------------------ save attention map (layer averaged) --------------------------------- # 
                        # CTRLNET
                        map_per_txt=[]
                        for txt_tkn_idx, id in enumerate(ids):
                            decoded_txt = tmp_tokenizer.decode([id])
                            map_per_layer=[]
                            for layer_idx, map1 in enumerate(ctrlnet_camaps):
                                n, d = map1.shape 
                                h, w = int(n**0.5), int(n**0.5)
                                vis_attn = map1[:, txt_tkn_idx].reshape(1,1,h,w)
                                vis_attn = F.interpolate(vis_attn, size=(512,512), mode='bilinear', align_corners=True) # 1 1 512 512 
                                vis_attn = (vis_attn-vis_attn.min())/(vis_attn.max()-vis_attn.min())
                                map_per_layer.append(vis_attn)
                            maps = torch.stack(map_per_layer)   # 7 1 1 512 512 
                            maps = maps.mean(dim=0)             # 1 1 512 512
                            maps = (maps-maps.min()) / (maps.max()-maps.min()) * 255.0
                            maps = maps.squeeze().detach().cpu().numpy().astype(np.uint8)   # 512 512 
                            heatmap = cv2.applyColorMap(maps, cv2.COLORMAP_JET)
                            gt_img = cv2.normalize(gt_img, None, 0, 255, cv2.NORM_MINMAX)
                            vis_img = (gt_img[:,:,::-1] + heatmap) / 2.0
                            cv2.putText(vis_img, decoded_txt, (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                            map_per_txt.append(vis_img)
                        hconcat_img1 = cv2.hconcat(map_per_txt)
                        # UNET
                        map_per_txt=[]
                        for txt_tkn_idx, id in enumerate(ids):
                            decoded_txt = tmp_tokenizer.decode([id])
                            map_per_layer=[]
                            for layer_idx, map1 in enumerate(unet_camaps):
                                n, d = map1.shape 
                                h, w = int(n**0.5), int(n**0.5)
                                vis_attn = map1[:, txt_tkn_idx].reshape(1,1,h,w)
                                vis_attn = F.interpolate(vis_attn, size=(512,512), mode='bilinear', align_corners=True) # 1 1 512 512 
                                vis_attn = (vis_attn-vis_attn.min())/(vis_attn.max()-vis_attn.min())
                                map_per_layer.append(vis_attn)
                            maps = torch.stack(map_per_layer)   # 7 1 1 512 512 
                            maps = maps.mean(dim=0)             # 1 1 512 512
                            maps = (maps-maps.min()) / (maps.max()-maps.min()) * 255.0
                            maps = maps.squeeze().detach().cpu().numpy().astype(np.uint8)   # 512 512 
                            heatmap = cv2.applyColorMap(maps, cv2.COLORMAP_JET)
                            gt_img = cv2.normalize(gt_img, None, 0, 255, cv2.NORM_MINMAX)
                            vis_img = (gt_img[:,:,::-1] + heatmap) / 2.0
                            cv2.putText(vis_img, decoded_txt, (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                            map_per_txt.append(vis_img)
                        hconcat_img2 = cv2.hconcat(map_per_txt)
                        concat_img = cv2.vconcat([hconcat_img1, hconcat_img2])
                        cv2.imwrite(f"{save_dir3}/ctrlnet_unet_{img_id}.jpg", concat_img)
                        # ------------------------------------ save attention map (layer averaged) --------------------------------- # 

                    if vis_args.all_attn_layers:
                        # ------------------------------------ save attention map (per layer) --------------------------------- # 
                        # CTRLNET
                        map_per_layer=[]
                        for layer_idx, map1 in enumerate(ctrlnet_camaps):
                            map_per_txt=[]
                            for txt_tkn_idx, id in enumerate(ids):
                                decoded_txt = tmp_tokenizer.decode([id])
                                n, d = map1.shape 
                                h, w = int(n**0.5), int(n**0.5)
                                vis_attn = map1[:, txt_tkn_idx].reshape(1,1,h,w)
                                vis_attn = F.interpolate(vis_attn, size=(512,512), mode='bilinear', align_corners=True) # 1 1 512 512 
                                vis_attn = (vis_attn-vis_attn.min())/(vis_attn.max()-vis_attn.min())*255.0
                                vis_attn = vis_attn.squeeze().detach().cpu().numpy().astype(np.uint8)
                                heatmap = cv2.applyColorMap(vis_attn, cv2.COLORMAP_JET)
                                gt_img = cv2.normalize(gt_img, None, 0, 255, cv2.NORM_MINMAX)
                                vis_img = (gt_img[:,:,::-1] + heatmap) / 2.0
                                cv2.putText(vis_img, decoded_txt, (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                                map_per_txt.append(vis_img)
                            map_per_txt_uint8 = [img.astype(np.uint8) for img in map_per_txt]
                            hconcat_img = cv2.hconcat(map_per_txt_uint8)
                            map_per_layer.append(hconcat_img)
                        vconcat_img = cv2.vconcat(map_per_layer)
                        cv2.imwrite(f"{save_dir1}/ctrlnet_{img_id}.jpg", vconcat_img)
                        # UNET
                        map_per_layer=[]
                        for map1 in unet_camaps:
                            map_per_txt=[]
                            for txt_tkn_idx, id in enumerate(ids):
                                decoded_txt = tmp_tokenizer.decode([id])
                                n, d = map1.shape 
                                h, w = int(n**0.5), int(n**0.5)
                                vis_attn = map1[:, txt_tkn_idx].reshape(1,1,h,w)
                                vis_attn = F.interpolate(vis_attn, size=(512,512), mode='bilinear', align_corners=True) # 1 1 512 512 
                                vis_attn = (vis_attn-vis_attn.min())/(vis_attn.max()-vis_attn.min())*255.0
                                vis_attn = vis_attn.squeeze().detach().cpu().numpy().astype(np.uint8)
                                heatmap = cv2.applyColorMap(vis_attn, cv2.COLORMAP_JET)
                                gt_img = cv2.normalize(gt_img, None, 0, 255, cv2.NORM_MINMAX)
                                vis_img = (gt_img[:,:,::-1] + heatmap) / 2.0
                                cv2.putText(vis_img, decoded_txt, (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
                                map_per_txt.append(vis_img)
                            map_per_txt_uint8 = [img.astype(np.uint8) for img in map_per_txt]
                            hconcat_img = cv2.hconcat(map_per_txt_uint8)
                            map_per_layer.append(hconcat_img)
                        vconcat_img = cv2.vconcat(map_per_layer)
                        cv2.imwrite(f"{save_dir2}/unet_{img_id}.jpg", vconcat_img)
                        # ------------------------------------ save attention map (per layer) --------------------------------- # 



            start2 = torch.cuda.Event(enable_timing=True)
            end2 = torch.cuda.Event(enable_timing=True)
            start2.record()
            # Text-spotting model forward pass 
            _, sampling_val_ocr_results = ts_model(extracted_feats, None, cfg.exp_args.mode)
            results_per_img = sampling_val_ocr_results[0]
            end2.record()
            torch.cuda.synchronize()
            ts_inf_time.append(start2.elapsed_time(end2))  # ms
            
            

            ts_pred_text=[]
            pred_polys=[]
            
            for j in range(len(results_per_img.polygons)):
                val_ctrl_pnt= results_per_img.polygons[j].view(16,2).cpu().detach().numpy().astype(np.int32)    # 32 -> 16 2
                val_rec = results_per_img.recs[j]
                val_pred_text = decode(val_rec)
                
                pred_polys.append(val_ctrl_pnt)
                ts_pred_text.append(val_pred_text)
            
            
            # ============================ Process with VLM ============================
            if cfg.prompter_args.use_vlm_prompt and i < cfg.vlm_args.inf_vlm_step:
                
                # select vlm input iamge 
                if cfg.vlm_args.vlm_input_img == 'LQ':
                    vlm_input_img = lq_img
                elif cfg.vlm_args.vlm_input_img == 'CLEAN':
                    vlm_input_img = cleaned_img
                elif cfg.vlm_args.vlm_input_img == 'RESTORE':
                    # pred_prev = torch.clamp((pure_cldm.vae_decode(x) + 1) / 2, min=0, max=1)
                    pred_x0 = torch.clamp((pure_cldm.vae_decode(pred_z0) + 1) / 2, min=0, max=1)
                    # save_image(pred_prev, './tmp_prev.jpg', normalize=True)
                    # save_image(pred_x0, './tmp_x0.jpg', normalize=True)
                    vlm_input_img = pred_x0
                    
                    
                # save input image for vlm
                vlm_img = TF.to_pil_image(vlm_input_img.squeeze(0).cpu().clamp(0, 1))  # Make sure shape is [3, H, W]
                tmp_path = f"./vlm_tmp/{cfg.exp_name}"
                os.makedirs(tmp_path, exist_ok=True)
                vlm_img_path = f'{tmp_path}/tmp.jpg'
                vlm_img.save(vlm_img_path)
                
                
                # set vlm usage 
                if cfg.vlm_args.vlm_text_correction:
                    pred_txt = [f'"{txt}"' for txt in ts_pred_text]
                    vlm_input_prompt =f"The image contains degraded or low-quality text. The OCR-predicted text may contain errors. Use both the visual appearance of the text and the predicted text to infer and correct the actual text. Only recognize and correct English text. OCR prediction: {', '.join(pred_txt) } Return only the corrected English text from the image."
                else:
                    vlm_input_prompt = f"OCR only the english texts in this image."
                vlm_results['vlm_input_prompt'] = vlm_input_prompt
                
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": vlm_img_path,
                            },
                            {"type": "text", 
                             "text": vlm_input_prompt}
                        ],
                    }
                ]


                # Preparation for inference
                text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = vlm_processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(vlm_model.device)
                # Inference: Generation of the output
                generated_ids = vlm_model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = vlm_processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                # print(output_text)


                # filter only english
                assert len(output_text) == 1, 'vlm output must be one'
                vlm_pred_text=[]
                for char in output_text[0]:
                    if 32 <= ord(char) and ord(char) < 127:
                        vlm_pred_text.append(char)
                vlm_pred_text=''.join(vlm_pred_text)
                
            # ============================ Process with VLM ============================
            
            
            # select prompt
            if cfg.prompter_args.use_gt_prompt:
                pred_texts = val_gt_text 
            elif cfg.prompter_args.use_ts_prompt:
                pred_texts = ts_pred_text 
            elif cfg.prompter_args.use_edit_prompt:
                pred_texts = [cfg.prompter_args.editting_text]
            elif cfg.prompter_args.use_vlm_prompt and i < cfg.vlm_args.inf_vlm_step:
                pred_texts = [vlm_pred_text]
            elif cfg.prompter_args.use_null_prompt:
                pred_texts = ['']
            else:
                pred_texts = ['']

            
            
            # select prompting style 
            pred_txt = [f'"{txt}"' for txt in pred_texts] 
            if cfg.prompter_args.prompt_style == 'CAPTION':
                pred_prompt = f"A realistic scene where the texts {', '.join(pred_txt) } appear clearly on signs, boards, buildings, or other objects."
            elif cfg.prompter_args.prompt_style == 'TAG':
                pred_prompt = f"{', '.join(pred_txt)}"
            

            # use pre computed vlm prompts 
            if cfg.prompter_args.use_llava_prompt or cfg.prompter_args.use_qwen_prompt:
                pred_prompt = kwargs['caption_anns'][img_id]['vlm_output']

            # override text condition with predicted prompt
            cond['c_txt'] = pure_cldm.clip.encode(pred_prompt)  # b 77 1024


            ts_results.append(
                dict(
                    timestep = current_timestep,
                    ts_pred_text = ts_pred_text,
                    pred_texts = pred_texts,
                    pred_prompt = pred_prompt,
                    pred_polys = pred_polys
                )
            )
        

        inf_time_img_avg = sum(img_inf_time) / len(img_inf_time) 
        inf_time_ts_avg = sum(ts_inf_time) / len(ts_inf_time)
        
        inf_time_modules['img_restoration_module'].append(inf_time_img_avg)
        inf_time_modules['text_spotting_module'].append(inf_time_ts_avg)

        return x, ts_results, vlm_results
    