# for debugging
# CUDA_VISIBLE_DEVICES=1 python train.py


# for ddp training
# python3 -m torch.distributed.launch --nproc_per_node=2 train.py



# cd experiment/deeplabv3_ddp_irn_bdcl_test
# python3 -m torch.distributed.launch --nproc_per_node=2 train.py
# python test.py