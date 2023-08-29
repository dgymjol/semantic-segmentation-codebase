# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import numpy as np
import random
torch.manual_seed(1) # cpu
torch.cuda.manual_seed_all(1) #gpu
np.random.seed(1) #numpy
random.seed(1) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import sys
import time

from config import config_dict
from datasets.generateData import generate_dataset
from net.generateNet import generate_net
import torch.optim as optim
from PIL import Image
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter 
from torch.utils.data import DataLoader, DistributedSampler
from net.sync_batchnorm.replicate import patch_replication_callback
from utils.configuration import Configuration
from utils.finalprocess import writelog
from utils.imutils import img_denorm
from net.sync_batchnorm import SynchronizedBatchNorm2d
from utils.visualization import generate_vis, max_norm
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

cfg = Configuration(config_dict)
def train_net():
    
	# 1. initialize process group
	dist.init_process_group("nccl")
	rank = dist.get_rank()
	torch.cuda.set_device(rank)
	device = torch.cuda.current_device()
	world_size = dist.get_world_size()

	period = 'train'
	transform = 'weak'
	dataset = generate_dataset(cfg, period=period, transform=transform)
	def worker_init_fn(worker_id):
		np.random.seed(1 + worker_id)
	
	sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=cfg.TRAIN_SHUFFLE)
	
	dataloader = DataLoader(dataset, 
				batch_size=cfg.TRAIN_BATCHES,
				sampler=sampler,
				shuffle=False,
				num_workers=cfg.DATA_WORKERS,
				pin_memory=True,
				drop_last=True,
				worker_init_fn=worker_init_fn)
	
	if cfg.GPUS > 1:
		# for DDP setting
  		print("++++++++++++++++++++ ", device)
  		net = generate_net(cfg, batchnorm=nn.SyncBatchNorm, dilated=cfg.MODEL_BACKBONE_DILATED, multi_grid=cfg.MODEL_BACKBONE_MULTIGRID, deep_base=cfg.MODEL_BACKBONE_DEEPBASE).cuda()
  		net = nn.parallel.DistributedDataParallel(net, device_ids=[device], output_device=device)
		# below line : only support in DP setting.
  		# net = generate_net(cfg, batchnorm=SynchronizedBatchNorm2d, dilated=cfg.MODEL_BACKBONE_DILATED, multi_grid=cfg.MODEL_BACKBONE_MULTIGRID, deep_base=cfg.MODEL_BACKBONE_DEEPBASE)
	
	else:
		net = generate_net(cfg, batchnorm=nn.BatchNorm2d, dilated=cfg.MODEL_BACKBONE_DILATED, multi_grid=cfg.MODEL_BACKBONE_MULTIGRID, deep_base=cfg.MODEL_BACKBONE_DEEPBASE)
	
	if cfg.TRAIN_CKPT:
		net.load_state_dict(torch.load(cfg.TRAIN_CKPT),strict=True)
		print('load pretrained model')
	if cfg.TRAIN_TBLOG:
		# Set the Tensorboard logger
		tblogger = SummaryWriter(cfg.LOG_DIR)	

	# print('Use %d GPU'%cfg.GPUS)
	# device = torch.device(0)
	if cfg.GPUS > 1:
	# 	net = nn.DataParallel(net)
	# 	patch_replication_callback(net)
		parameter_source = net.module
	else:
		parameter_source = net
	# net.to(device)		
	criterion = nn.CrossEntropyLoss(ignore_index=255)
	optimizer = optim.SGD(
		params = [
			{'params': get_params(parameter_source,key='backbone'), 'lr': cfg.TRAIN_LR},
			{'params': get_params(parameter_source,key='cls'),      'lr': 10*cfg.TRAIN_LR},
			{'params': get_params(parameter_source,key='others'),   'lr': cfg.TRAIN_LR}
		],
		momentum=cfg.TRAIN_MOMENTUM,
		weight_decay=cfg.TRAIN_WEIGHT_DECAY
	)
	itr = cfg.TRAIN_MINEPOCH * len(dataset)//(cfg.TRAIN_BATCHES)
	max_itr = cfg.TRAIN_ITERATION
	max_epoch = max_itr*(cfg.TRAIN_BATCHES)//len(dataset)+1
	criterion = nn.CrossEntropyLoss(ignore_index=255)
	scaler = torch.cuda.amp.GradScaler()
	with tqdm(total=max_itr) as pbar:
		for epoch in range(cfg.TRAIN_MINEPOCH, max_epoch):
			for i_batch, sample in enumerate(dataloader):

				now_lr = adjust_lr(optimizer, itr, max_itr, cfg.TRAIN_LR, cfg.TRAIN_POWER)
				optimizer.zero_grad()

				inputs, seg_label = sample['image'].cuda(), sample['segmentation'].cuda()
				n,c,h,w = inputs.size()

				with torch.cuda.amp.autocast():
					pred1 = net(inputs)
					loss = criterion(pred1, seg_label)
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()

				pbar.set_description("loss=%g " % (loss.item()))
				pbar.update(1)
				time.sleep(0.001)
				# print('epoch:%d/%d\tbatch:%d/%d\titr:%d\tlr:%g\tloss:%g' % 
				# 	(epoch, max_epoch, i_batch, len(dataset)//(cfg.TRAIN_BATCHES),
				# 	itr+1, now_lr, loss.item()))
				if cfg.TRAIN_TBLOG and itr%100 == 0 and device == 0:
					inputs1 = img_denorm(inputs[-1].cpu().numpy()).astype(np.uint8)
					label1 = sample['segmentation'][-1].cpu().numpy()
					label_color1 = dataset.label2colormap(label1).transpose((2,0,1))

					n,c,h,w = inputs.size()
					seg_vis1 = torch.argmax(pred1[-1], dim=0).detach().cpu().numpy()
					seg_color1 = dataset.label2colormap(seg_vis1).transpose((2,0,1))

					tblogger.add_scalar('loss', loss.item(), itr)
					tblogger.add_scalar('lr', now_lr, itr)
					tblogger.add_image('Input', inputs1, itr)
					tblogger.add_image('Label', label_color1, itr)
					tblogger.add_image('SEG1', seg_color1, itr)
				itr += 1
				if itr>=max_itr:
					break
			save_path = os.path.join(cfg.MODEL_SAVE_DIR,'%s_%s_%s_epoch%d.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,epoch))
			torch.save(parameter_source.state_dict(), save_path)
			print('%s has been saved'%save_path)
			# remove_path = os.path.join(cfg.MODEL_SAVE_DIR,'%s_%s_%s_epoch%d.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,epoch-1))
			# if os.path.exists(remove_path):
			# 	os.remove(remove_path)
			
	save_path = os.path.join(cfg.MODEL_SAVE_DIR,'%s_%s_%s_itr%d_all.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,cfg.TRAIN_ITERATION))
	torch.save(parameter_source.state_dict(),save_path)
	if cfg.TRAIN_TBLOG and device == 0 :
		tblogger.close()
	print('%s has been saved'%save_path)
	writelog(cfg, period)

def adjust_lr(optimizer, itr, max_itr, lr_init, power):
	now_lr = lr_init * (1 - itr/(max_itr+1)) ** power
	optimizer.param_groups[0]['lr'] = now_lr
	optimizer.param_groups[1]['lr'] = 10*now_lr
	optimizer.param_groups[2]['lr'] = now_lr
	return now_lr

def get_params(model, key):
	for m in model.named_modules():
		if key == 'backbone':
			if ('backbone' in m[0]) and isinstance(m[1], (nn.Conv2d, SynchronizedBatchNorm2d, nn.BatchNorm2d, nn.InstanceNorm2d)):
				for p in m[1].parameters():
					yield p
		elif key == 'cls':
			if ('cls_conv' in m[0]) and isinstance(m[1], (nn.Conv2d, SynchronizedBatchNorm2d, nn.BatchNorm2d, nn.InstanceNorm2d)):
				for p in m[1].parameters():
					yield p
		elif key == 'others':
			if ('backbone' not in m[0] and 'cls_conv' not in m[0]) and isinstance(m[1], (nn.Conv2d, SynchronizedBatchNorm2d, nn.BatchNorm2d, nn.InstanceNorm2d)):
				for p in m[1].parameters():
					yield p
if __name__ == '__main__':
	train_net()


