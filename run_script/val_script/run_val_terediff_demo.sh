
CUDA_VISIBLE_DEVICES=7 accelerate launch val_demo.py         --config configs/val/val_terediff_demo.yaml \
                                                        --config_testr testr/configs/TESTR/TESTR_R_50_Polygon.yaml \
