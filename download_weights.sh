mkdir weights
cd weights

wget https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/realesrgan_s4_swinir_100k.pth
wget https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/DiffBIR_v2.1.pt
wget https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/sd2.1-base-zsnr-laionaes5.ckpt

cd ..
mkdir model_ckpts 
cd model_ckpts
gdown 14qtLOso_kurfY_FOzOWUR8z-_IvRwy-X

