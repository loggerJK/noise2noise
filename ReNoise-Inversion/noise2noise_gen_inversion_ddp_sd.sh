export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1

torchrun --nproc_per_node=4  /media/dataset1/project/jiwon/noise2noise/ReNoise-Inversion/noise2noise_gen_inversion_ddp_sd.py --name coco_latents_sd2.1
# python -m pdb  /media/dataset1/project/jiwon/ReNoise-Inversion/noise2noise_gen_inversion_ddp_sd.py --name TEST