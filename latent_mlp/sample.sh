export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=""

python sample.py --target_size 30 --model_path /media/dataset2/jiwon/noise2noise/latent_mlp/results/unet_upscaling_reg_full/model_195epoch.pt


