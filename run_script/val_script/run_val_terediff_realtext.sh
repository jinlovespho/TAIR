
CUDA_VISIBLE_DEVICES=6 accelerate launch val.py         --config configs/val/val_terediff_realtext.yaml \
                                                        --config_testr testr/configs/TESTR/TESTR_R_50_Polygon.yaml \
