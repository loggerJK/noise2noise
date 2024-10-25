export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4  /media/dataset2/jiwon/noise2noise/ReNoise-Inversion/noise2noise_gen_inversion_ddp_sd.py --name new_coco
# python -m pdb  /media/dataset1/project/jiwon/ReNoise-Inversion/noise2noise_gen_inversion_ddp_sd.py --name TEST