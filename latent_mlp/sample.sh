export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

python sample.py --target_size 30 --model_path /media/dataset1/project/jiwon/noise2noise/latent_mlp/results/upscaling_reg_full/model_2epoch.pt



