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

from collections import deque

from torch.nn.functional import cosine_similarity


cfg = Configuration(config_dict)

class BoundaryLoss(nn.Module):
	def __init__(self, temperature=1.0, memory_size=1000, embedding_dim=128, margin=0.2):
		super(BoundaryLoss, self).__init__()
		self.temperature = temperature
		self.memory_size = memory_size
		self.register_buffer('pos_memory', torch.randn(memory_size, embedding_dim))
		self.register_buffer('neg_memory', torch.randn(memory_size, embedding_dim))
		self.pos_memory_ptr = 0
		self.neg_memory_ptr = 0
		self.margin = margin

	def update_pos_memory(self, embeddings):
		num_embed = embeddings.size(0)

		if self.pos_memory_ptr + num_embed <= self.memory_size:
			self.pos_memory[self.pos_memory_ptr:self.pos_memory_ptr + num_embed] = embeddings.detach()
		else:			
			num_possible = self.memory_size-self.pos_memory_ptr
			num_remain = num_embed - num_possible
			self.pos_memory[self.pos_memory_ptr:] = embeddings.detach()[:num_possible]
			self.pos_memory[:num_remain] = embeddings.detach()[num_possible:]
   
		self.pos_memory_ptr = (self.pos_memory_ptr + num_embed) % self.memory_size
 
	def update_neg_memory(self, embeddings):
		num_embed = embeddings.size(0)
  
		if self.neg_memory_ptr + num_embed <= self.memory_size:
			self.neg_memory[self.neg_memory_ptr:self.neg_memory_ptr + num_embed] = embeddings.detach()
		else:			
			num_possible = self.memory_size-self.neg_memory_ptr
			num_remain = num_embed - num_possible
			self.neg_memory[self.neg_memory_ptr:] = embeddings.detach()[:num_possible]
			self.neg_memory[:num_remain] = embeddings.detach()[num_possible:]
   
		self.neg_memory_ptr = (self.neg_memory_ptr + num_embed) % self.memory_size
  
	# def get_gt_mask(self, dataset, names, feat_shape):
	# 	gts = []

	# 	for name in names:
	# 		seg_file = dataset.seg_dir + '/' + name + '.png'
	# 		gts.append(torch.tensor(np.array(Image.open(seg_file))))
   
	# 	breakpoint()
	# 	gts = torch.stack(gts, dim=0).unsqueeze(1).float()
	# 	# .unsqueeze(1).float()
	# 	# s_gt = F.interpolate(gt, feat_shape, mode='nearest').squeeze(1)
	# 	# gt_mask = (s_gt.reshape(-1)) != 0
	# 	# gt_masks.append(gt_mask)
   
	# 	return torch.cat(gt_masks, dim=0)
 
	def sampling_mask(self, original_mask, num_samples):
		if torch.sum(original_mask) > num_samples:
			true_indices = torch.nonzero(original_mask).squeeze()
			shuffled_indices = torch.randperm(true_indices.size(0))[:num_samples]
			selected_true_indices = true_indices[shuffled_indices]

			new_mask = torch.zeros_like(original_mask)
			new_mask[selected_true_indices] = True

			return new_mask
		else:
			return original_mask
		
	def forward(self, preds, embeddings, fsss_gts):
		# preds.shape : (B, C, H, W)
		# embeddings : (B, embed_dim, 64, 64)
		# fsss_gts.shape : (B, H, W)

		embeddings = embeddings.clone()
		feat_shape = (embeddings.shape[2], embeddings.shape[3])

		# Make pred shape same as embeddings >> preds(B, C, H, W) -> s_pred(B, 64, 64)
		preds_clone = preds.clone().argmax(dim=1).unsqueeze(1).float()
		s_pred = F.interpolate(preds_clone, feat_shape, mode='nearest').squeeze(1)
	
		# Make FSSS_GT shape same as embeddings >> fsss_gts(B, H, W) -> s_fsss_gt(B, 64, 64)
		fsss_gts_clone = fsss_gts.clone().unsqueeze(1).float()
		s_fsss_gt = F.interpolate(fsss_gts_clone, feat_shape, mode='nearest').squeeze(1)
  
		# Normalize embeddings
		embeddings = F.normalize(embeddings.clone(), p=2, dim=1)
  
		# Make 2D -> 1D (num_pixels = B * 4096)
		s_pred = s_pred.reshape(-1) #(B, 4096)
		s_fsss_gt = s_fsss_gt.reshape(-1)
		embeddings = embeddings.permute(1, 0, 2, 3).reshape(embeddings.shape[1], -1).t() #(B * 4096, 256)

		# Make pred mask and fsss_gt_mask (foreground = True, background = False)
		pred_mask = (s_pred!=0)
		fsss_gt_mask = ((s_fsss_gt != 0) & (s_fsss_gt != 255))
		ignore_mask = (s_fsss_gt == 255)

		# Calculate mask for anchor, positive and negative samples
		anchor_mask = (pred_mask & fsss_gt_mask)
		positive_mask = (fsss_gt_mask ^ anchor_mask)
		negative_mask = (pred_mask ^ ((fsss_gt_mask | ignore_mask) & pred_mask))
  
		# sampling masks
		new_anc_mask = self.sampling_mask(anchor_mask, self.memory_size // 10)
		new_pos_mask = self.sampling_mask(positive_mask, self.memory_size // 3)
		new_neg_mask = self.sampling_mask(negative_mask, self.memory_size // 3)
  
		# Update memory with current batch embeddings
		self.update_pos_memory(embeddings[new_pos_mask])
		self.update_neg_memory(embeddings[new_neg_mask])
  
		# Calculate anchor-positive cosine similarities and anchor-negative cosine similarities
		anc_embeds = embeddings[new_anc_mask]
		losses = torch.zeros(len(anc_embeds), self.memory_size ** 2).cuda()

		for idx, anc_embed in enumerate(anc_embeds):
			pos_similarity = cosine_similarity(anc_embed.repeat(self.memory_size, 1), self.pos_memory.clone().cuda())
			neg_similarity = cosine_similarity(anc_embed.repeat(self.memory_size, 1), self.neg_memory.clone().cuda())
			losses[idx] = (pos_similarity.repeat(self.memory_size).reshape(self.memory_size, -1).t().reshape(-1) - neg_similarity.repeat(self.memory_size) + self.margin)

		# Calculate triplet loss
		loss = F.relu(losses.reshape(-1)).mean()
  
		return loss

class SupConLoss(nn.Module):
	def __init__(self, temperature=1.0, memory_size=1000, embedding_dim=128, margin=0.2, num_classes=21, hard_ratio=0.8, views=3):
		super(SupConLoss, self).__init__()
		self.temperature = temperature
		self.memory_size = memory_size
		self.register_buffer('memory', torch.randn(num_classes-1, memory_size, embedding_dim))
		self.memory_ptr = [0] * (num_classes-1)
		self.margin = margin
		self.num_classes = num_classes
		self.hard_ratio = hard_ratio
		self.views = views
		self.epsilon = 1e-6

	def update_memory(self, embeddings, cls):
		num_embed = embeddings.size(0)

		cls_ptr = self.memory_ptr[cls-1]
		if cls_ptr + num_embed <= self.memory_size:
			self.memory[cls-1][cls_ptr:cls_ptr + num_embed] = embeddings.detach()
		else:			
			num_possible = self.memory_size-cls_ptr
			num_remain = num_embed - num_possible
			self.memory[cls-1][cls_ptr:] = embeddings.detach()[:num_possible]
			self.memory[cls-1][:num_remain] = embeddings.detach()[num_possible:]
   
		self.memory_ptr[cls-1] = (cls_ptr + num_embed) % self.memory_size

	def sampling_mask(self, original_mask, num_samples):
		if torch.sum(original_mask) > num_samples:
			true_indices = torch.nonzero(original_mask).squeeze()
			shuffled_indices = torch.randperm(true_indices.size(0))[:num_samples]
			selected_true_indices = true_indices[shuffled_indices]

			new_mask = torch.zeros_like(original_mask)
			new_mask[selected_true_indices] = True

			return new_mask
		else:
			return original_mask
		
	def forward(self, preds, embeddings, wsss_gts, fsss_gts):
		# preds.shape : (B, C, H, W)
		# embeddings : (B, embed_dim, 64, 64)
		# wsss_gts.shape : (B, H, W) 
		# fsss_gts.shape : (B, H, W)

		embeddings = embeddings.clone()
		feat_shape = (embeddings.shape[2], embeddings.shape[3])
  
		# Make WSSS_GT shape same as embeddings >> wsss_gts(B, H, W) -> s_gt(B, 64, 64)
		wsss_gts_clone = wsss_gts.clone().unsqueeze(1).float()
		s_wsss_gt = F.interpolate(wsss_gts_clone, feat_shape, mode='nearest').squeeze(1)

		# Make pred shape same as embeddings >> preds(B, C, H, W) -> s_pred(B, 64, 64)
		preds_clone = preds.clone().argmax(dim=1).unsqueeze(1).float()
		s_pred = F.interpolate(preds_clone, feat_shape, mode='nearest').squeeze(1)
	
		# Make FSSS_GT shape same as embeddings >> fsss_gts(B, H, W) -> s_fsss_gt(B, 64, 64)
		fsss_gts_clone = fsss_gts.clone().unsqueeze(1).float()
		s_fsss_gt = F.interpolate(fsss_gts_clone, feat_shape, mode='nearest').squeeze(1)
  
		# Normalize embeddings
		embeddings = F.normalize(embeddings.clone(), p=2, dim=1)
  
		# Make 2D -> 1D (num_pixels = B * 4096)
		s_wsss_gt = s_wsss_gt.reshape(-1) #(B, 4096)
		s_pred = s_pred.reshape(-1) #(B, 4096)
		s_fsss_gt = s_fsss_gt.reshape(-1)
		embeddings = embeddings.permute(1, 0, 2, 3).reshape(embeddings.shape[1], -1).t() #(B * 4096, 256)

		# Make pred mask and fsss_gt_mask (foreground = True, background = False)
  
		fg_fsss_gt_mask = ((s_fsss_gt != 0) & (s_fsss_gt != 255))
		s_wsss_cls = torch.unique(s_wsss_gt[((s_wsss_gt != 0) & (s_wsss_gt != 255))].type(torch.uint8)).tolist()
  
  
		pos_similarities, neg_similarities = [], []
		anchor_maskes = []
		losses = []
  
		for cls in s_wsss_cls: # except background class
			cls = s_wsss_cls[0]
			cls_pred_mask = (s_pred == cls)
			cls_wsss_gt_mask = (s_wsss_gt == cls)
			
			# Calculate mask for anchor, positive and negative samples
			cls_anchor_mask = (cls_wsss_gt_mask & fg_fsss_gt_mask)
			easy_cls_anchor_mask = (cls_anchor_mask) & (cls_pred_mask) # easy = pred correct
			hard_cls_anchor_mask = (cls_anchor_mask) & (~cls_pred_mask) # hard = pred hard

			# sampling masks
			new_easy_cls_anchor_mask = self.sampling_mask(easy_cls_anchor_mask, int((self.memory_size // 5) * (1-self.hard_ratio)))
			new_hard_cls_anchor_mask = self.sampling_mask(hard_cls_anchor_mask, int((self.memory_size // 5) * self.hard_ratio))
			new_cls_anchor_mask = new_easy_cls_anchor_mask | new_hard_cls_anchor_mask
			anchor_maskes.append((cls, new_cls_anchor_mask))

			# Calculate anchor-positive cosine similarities and anchor-negative cosine similarities
			cls_anchor_embeds = embeddings[new_cls_anchor_mask]

			if len(cls_anchor_embeds) == 0:
				return 0

			for cls_anc_embed in cls_anchor_embeds:
				all_neg_similarities = []
				for i in range(self.num_classes-1):
					if i == cls - 1:
						pos_similarity = cosine_similarity(cls_anc_embed.repeat(self.memory_size, 1), self.memory[i].clone().cuda())
					else:
						neg_similarity = cosine_similarity(cls_anc_embed.repeat(self.memory_size, 1), self.memory[i].clone().cuda())
						all_neg_similarities.append(neg_similarity)
		
				all_neg_similarities = torch.cat(all_neg_similarities, dim=0)
				hard_neg_similarity = torch.sort(all_neg_similarities)[0][-self.memory_size:]

				losses.append(pos_similarity.repeat(self.memory_size).reshape(self.memory_size, -1).t().reshape(-1) - hard_neg_similarity.repeat(self.memory_size) + self.margin)

		# Calculate triplet loss
		loss = F.relu(torch.cat(losses, dim=0).cuda()).mean()

		# Update memory with current batch embeddings
		for cls, anchor_mask in anchor_maskes:
			self.update_memory(embeddings[anchor_mask], cls)
   
		return loss


def worker_init_fn(worker_id):
	np.random.seed(1 + worker_id)
 
def train_net():

	period = 'train'
	transform = 'weak'
	dataset = generate_dataset(cfg, period=period, transform=transform)

	if cfg.GPUS > 1:
		# for DDP setting

		# 1. initialize process group
		dist.init_process_group("nccl")
		rank = dist.get_rank()
		torch.cuda.set_device(rank)
		device = torch.cuda.current_device()
		world_size = dist.get_world_size()
		print("++++++++++++++++++++ ", device)

		sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=cfg.TRAIN_SHUFFLE)

		dataloader = DataLoader(dataset, 
					batch_size=cfg.TRAIN_BATCHES,
					sampler=sampler,
					shuffle=False,
					num_workers=cfg.DATA_WORKERS,
					pin_memory=True,
					drop_last=True,
					worker_init_fn=worker_init_fn)

		net = generate_net(cfg, batchnorm=nn.SyncBatchNorm, dilated=cfg.MODEL_BACKBONE_DILATED, multi_grid=cfg.MODEL_BACKBONE_MULTIGRID, deep_base=cfg.MODEL_BACKBONE_DEEPBASE).cuda()
		net = nn.parallel.DistributedDataParallel(net, device_ids=[device], output_device=device)
		# below line : only support in DP setting.
		# net = generate_net(cfg, batchnorm=SynchronizedBatchNorm2d, dilated=cfg.MODEL_BACKBONE_DILATED, multi_grid=cfg.MODEL_BACKBONE_MULTIGRID, deep_base=cfg.MODEL_BACKBONE_DEEPBASE)
		parameter_source = net.module

	else:
		dataloader = DataLoader(dataset, 
					batch_size=cfg.TRAIN_BATCHES, 
					shuffle=cfg.TRAIN_SHUFFLE, 
					num_workers=cfg.DATA_WORKERS,
					pin_memory=True,
					drop_last=True,
					worker_init_fn=worker_init_fn)

		device = 0
		net = generate_net(cfg, batchnorm=nn.BatchNorm2d, dilated=cfg.MODEL_BACKBONE_DILATED, multi_grid=cfg.MODEL_BACKBONE_MULTIGRID, deep_base=cfg.MODEL_BACKBONE_DEEPBASE).cuda()
		parameter_source = net

	if cfg.TRAIN_CKPT:
		net.load_state_dict(torch.load(cfg.TRAIN_CKPT),strict=True)
		print('load pretrained model')
	if cfg.TRAIN_TBLOG:
		# Set the Tensorboard logger
		tblogger = SummaryWriter(cfg.LOG_DIR)	

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

	ce_fn = nn.CrossEntropyLoss(ignore_index=255)
	bdry_fn = BoundaryLoss(temperature=cfg.TEMP, memory_size=cfg.BD_MEMORY_SIZE, embedding_dim=cfg.EMBEDDING_DIM, margin=cfg.TRIPLET_MARGIN)
	supcon_fn = SupConLoss(temperature=cfg.TEMP, memory_size=cfg.SUPCON_MEMORY_SIZE, embedding_dim=cfg.EMBEDDING_DIM, margin=cfg.TRIPLET_MARGIN, num_classes=cfg.MODEL_NUM_CLASSES, hard_ratio=cfg.HARD_RATIO)

	# if cfg.ENABLE_MEMORY:
	# 	memory_bank = MemoryBank(cfg.TRAIN_BATCHES * 4)

	scaler = torch.cuda.amp.GradScaler()
	with tqdm(total=max_itr) as pbar:
		for epoch in range(cfg.TRAIN_MINEPOCH, max_epoch):
			for i_batch, sample in enumerate(dataloader):

				now_lr = adjust_lr(optimizer, itr, max_itr, cfg.TRAIN_LR, cfg.TRAIN_POWER)
				optimizer.zero_grad()

				inputs, seg_label = sample['image'].cuda(), sample['segmentation'].cuda()
				n,c,h,w = inputs.size()

				with torch.cuda.amp.autocast():
					outputs = net(inputs)
					pred, feat = outputs['pred_seg'], outputs['embed']

					ce_loss = ce_fn(pred, seg_label)
					# bdry_loss = 0
					# supcon_loss = 0
     
					if itr >= cfg.CONTRAST_WARMUP:
						# Calculate triplet loss using boundary information

						if cfg.BDRY_LOSS:
							bdry_loss = bdry_fn(pred, feat, sample['gt'].cuda())
							bdry_loss *= cfg.BDRY_LOSS_WEIGHT
							print('boundary_loss : ', bdry_loss)
       
						if cfg.SUPCON_LOSS:
							supcon_loss = supcon_fn(pred, feat, seg_label, sample['gt'].cuda())
							supcon_loss *= cfg.SUPCON_LOSS_WEIGHT
							print('boundary_loss : ', supcon_loss)

       
					else:
						bdry_loss = 0 * (bdry_fn(pred, feat, sample['gt'].cuda()))
						supcon_loss = 0 * (supcon_fn(pred, feat, seg_label, sample['gt'].cuda()))
						# bdry_loss = 0
						# supcon_loss = 0 
      

					loss = ce_loss + bdry_loss + supcon_loss
     
				# with torch.cuda.amp.autocast():
				# 	outputs = net(inputs)
				# 	pred, feat = outputs['pred_seg'], outputs['embed']

				# 	ce_loss = ce_fn(pred, seg_label)
				# 	bdry_loss = 0
				# 	supcon_loss = 0
				# 	# ce_loss_weight = 1
     
				# 	if itr >= cfg.CONTRAST_WARMUP:
				# 		# Calculate triplet loss using boundary information

				# 		if cfg.BDRY_LOSS:
				# 			bdry_loss = bdry_fn(pred, feat, sample['gt'].cuda())
				# 			bdry_loss *= cfg.BDRY_LOSS_WEIGHT
				# 			# ce_loss_weight -= cfg.BDRY_LOSS_WEIGHT
				# 		# else:
				# 		# 	bdry_loss = 0 * bdry_fn(pred, feat, seg_label, sample['gt'].cuda())

				# 		if cfg.SUPCON_LOSS:
				# 			supcon_loss = supcon_fn(pred, feat, seg_label, sample['gt'].cuda())
				# 			supcon_loss *= cfg.SUPCON_LOSS_WEIGHT
				# 			# ce_loss_weight -= cfg.SUPCON_LOSS_WEIGHT
				# 		# else:
				# 		# 	supcon_loss = 0 * supcon_fn(pred, feat, seg_label, sample['gt'].cuda())
       
				# 	loss = ce_loss + bdry_loss + supcon_loss
		
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
					seg_vis1 = torch.argmax(pred[-1], dim=0).detach().cpu().numpy()
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


