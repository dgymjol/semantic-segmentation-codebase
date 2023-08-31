# for debugging
CUDA_VISIBLE_DEVICES=1 python train.py


# for ddp training
python3 -m torch.distributed.launch --nproc_per_node=2 train.py