
CUDA_VISIBLE_DEVICES=3 accelerate launch val.py         --config configs/val/val_terediff_satext_lv1.yaml \
                                                        --config_testr testr/configs/TESTR/TESTR_R_50_Polygon.yaml \
