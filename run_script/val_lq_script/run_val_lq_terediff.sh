
CUDA_VISIBLE_DEVICES=2 accelerate launch val_lq.py      --config configs/val_lq/val_terediff.yaml \
                                                        --config_testr testr/configs/TESTR/TESTR_R_50_Polygon.yaml \
