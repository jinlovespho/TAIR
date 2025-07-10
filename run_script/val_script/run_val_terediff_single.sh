
CUDA_VISIBLE_DEVICES=6 accelerate launch val_single.py      --config configs/val/val_terediff_single.yaml \
                                                            --config_testr testr/configs/TESTR/TESTR_R_50_Polygon.yaml \
