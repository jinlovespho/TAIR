
CUDA_VISIBLE_DEVICES=1 accelerate launch val_textzoom.py         --config configs/val/val_terediff_textzoom.yaml \
                                                                --config_testr testr/configs/TESTR/TESTR_R_50_Polygon.yaml \
