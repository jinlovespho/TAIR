
CUDA_VISIBLE_DEVICES=3,4,5,6 accelerate launch train.py       --config configs/train/train_stage1_terediff.yaml \
                                                        --config_testr testr/configs/TESTR/TESTR_R_50_Polygon.yaml \
