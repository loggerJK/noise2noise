export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

python sample_dm_cond.py --target_size 30 --model_path /media/dataset2/jiwon/noise2noise/latent_mlp/results/unet_dm_cond/model_30epoch



