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

# class MemoryBank:
# 	def __init__(self, memory_size: int):
# 		self.memory_size = memory_size
# 		self.bank = deque(maxlen=self.memory_size)

# 	def append(self, pred, feat, gt):
# 		pred, feat, gt = pred.cpu().detach(), feat.cpu().detach(), gt.cpu().detach()

# 		for pred_cls, feature, gt_cls in zip(pred, feat, gt):
# 			entry = {"feature": feature, "pred_cls": pred_cls, "gt_cls" : gt_cls}
# 			self.bank.append(entry)

# def get_bdry_anc_pos_neg(dataset, names, preds, feats):
# 	gt_bdrys = []

# 	for name in names:
# 		seg_file = dataset.seg_dir + '/' + name + '.png'
# 		segmentation = np.array(Image.open(seg_file))
# 		gt_bdry = segmentation!=0
# 		gt_bdrys.append(gt_bdry)


# 	return gt_bdrys

# def compute_triplet_loss(anc, pos, neg, epsilon=1e-6):
# 	return None

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
		
	def forward(self, preds, embeddings, gts, fsss_gts):
		# preds.shape : (B, C, H, W)
		# embeddings : (B, embed_dim, 64, 64)
		# gts.shape : (B, H, W)     * wsss gt
		# fsss_gts.shape : (B, H, W)

		embeddings = embeddings.clone()
		feat_shape = (embeddings.shape[2], embeddings.shape[3])
  
		# Make GT shape same as embeddings >> gts(B, H, W) -> s_gt(B, 64, 64)
		gts_clone = gts.clone().unsqueeze(1).float()
		s_gt = F.interpolate(gts_clone, feat_shape, mode='nearest').squeeze(1)

		# Make pred shape same as embeddings >> preds(B, C, H, W) -> s_pred(B, 64, 64)
		preds_clone = preds.clone().argmax(dim=1).unsqueeze(1).float()
		s_pred = F.interpolate(preds_clone, feat_shape, mode='nearest').squeeze(1)
	
		# Make FSSS_GT shape same as embeddings >> fsss_gts(B, H, W) -> s_fsss_gt(B, 64, 64)
		fsss_gts_clone = fsss_gts.clone().unsqueeze(1).float()
		s_fsss_gt = F.interpolate(fsss_gts_clone, feat_shape, mode='nearest').squeeze(1)
  
		# Normalize embeddings
		embeddings = F.normalize(embeddings.clone(), p=2, dim=1)
  
		# Make 2D -> 1D (num_pixels = B * 4096)
		s_gt = s_gt.reshape(-1) #(B, 4096)
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
		pos_similarities, neg_similarities = [], []
		for anc_embed in anc_embeds:
			pos_similarity = torch.matmul(anc_embed, self.pos_memory.clone().cuda().t())
			neg_similarity = torch.matmul(anc_embed, self.neg_memory.clone().cuda().t())
			pos_similarities.append(pos_similarity)
			neg_similarities.append(neg_similarity)

		pos_similarities = torch.cat(pos_similarities, dim=0).mean()
		neg_similarities = torch.cat(neg_similarities, dim=0).mean()

		# Calculate triplet loss
		loss = F.relu(pos_similarities.clone() - neg_similarities.clone() + self.margin).clone()

		return loss

# class ContrastiveLossWithMemory(nn.Module):
# 	def __init__(self, temperature=1.0, memory_size=1000, embedding_dim=128):
# 		super(ContrastiveLossWithMemory, self).__init__()
# 		self.temperature = temperature
# 		self.memory_size = memory_size
# 		self.register_buffer('memory', torch.randn(memory_size, embedding_dim))
# 		self.register_buffer('memory_labels', torch.zeros(memory_size, dtype=torch.long))
# 		self.register_buffer('memory_preds', torch.zeros(memory_size, dtype=torch.long))
# 		self.memory_ptr = 0

# 	def update_memory(self, embeddings, predicted_classes, gt_classes):
# 		batch_size = embeddings.size(0)
# 		self.memory[self.memory_ptr:self.memory_ptr + batch_size] = embeddings.detach()
# 		self.memory_labels[self.memory_ptr:self.memory_ptr + batch_size] = gt_classes
# 		self.memory_preds[self.memory_ptr:self.memory_ptr + batch_size] = predicted_classes
# 		self.memory_ptr = (self.memory_ptr + batch_size) % self.memory_size

# 	def forward(self, preds, embeddings, gts, num_batches, hard_neg=True):
# 		# preds.shape : (B, C, H, W)
# 		# embeddings : (B, embed_dim, 64, 64)
# 		# gts.shape : (B, H, W)
  
# 		breakpoint()
		
# 		batch_size = embeddings.shape[0]
# 		feat_shape = (embeddings.shape[2], embeddings.shape[3])
  
# 		# Make GT shape same as embeddings >> gts(B, H, W) -> s_gt(B, 64, 64)
# 		gts_clone = gts.clone().unsqueeze(1).float()
# 		s_gt = F.interpolate(gts_clone, feat_shape, mode='nearest').squeeze(1)

# 		# Make pred shape same as embeddings >> preds(B, C, H, W) -> s_pred(B, 64, 64)
# 		preds_clone = preds.clone().argmax(dim=1).unsqueeze(1).float()
# 		s_pred = F.interpolate(preds_clone, feat_shape, mode='nearest').squeeze(1)
	
# 		# Normalize embeddings
# 		embeddings = F.normalize(embeddings, p=2, dim=1)
  
# 		# Make 2D -> 1D (num_pixels = B * 4096)
# 		s_gt = s_gt.reshape(batch_size, -1) #(B * 4096)
# 		s_pred = s_pred.reshape(batch_size, -1) #(B * 4096)
# 		embeddings = embeddings.permute(1, 0, 2, 3).reshape(embeddings.shape[1], -1).t() #(B * 4096, 256)


# 		# Update memory with current batch embeddings
# 		self.update_memory(embeddings, preds, gts)

# 		# Calculate cosine similarity between embeddings and memory
# 		similarity_matrix = torch.matmul(embeddings, self.memory.t()) / self.temperature

# 		# Create positive mask (same predicted classes in memory)
# 		positive_mask = torch.eq(gts, self.memory_labels)
  
# 		# Create negative mask (different predicted classes in memory)
# 		negative_mask = ~positive_mask
# 		# if hard_neg:
# 		# 	true_pred_mask = torch.eq(preds, gts)
# 		# 	false_pred_mask = ~true_pred_mask
# 		# 	hard_negative_mask = (negative_mask & true_pred_mask)
  
# 		# Calculate loss for positive pairs
# 		exp_sim_positive = torch.exp(similarity_matrix) * positive_mask.float()
# 		pos_exp_sum = exp_sim_positive.sum(1, keepdim=True)
# 		positive_loss = -torch.log(exp_sim_positive / pos_exp_sum).sum(1)

# 		# Calculate loss for negative pairs 
# 		exp_sim_negative = torch.exp(similarity_matrix) * negative_mask.float()
# 		neg_exp_sum = exp_sim_negative.sum(1, keepdim=True)
# 		negative_loss = -torch.log(1.0 - exp_sim_negative / neg_exp_sum).sum(1)

# 		# Combine positive and negative losses
# 		loss = positive_loss + negative_loss

# 		return loss.mean()


cfg = Configuration(config_dict)

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
	bdry_fn = BoundaryLoss(temperature=cfg.TEMP, memory_size=cfg.MEMORY_SIZE, embedding_dim=cfg.EMBEDDING_DIM, margin=cfg.TRIPLET_MARGIN)

	# if cfg.ENABLE_MEMORY:
	# 	memory_bank = MemoryBank(cfg.TRAIN_BATCHES * 4)
	# supcon_fn = ContrastiveLossWithMemory(temperature=cfg.TEMP, memory_size=(cfg.TRAIN_BATCHES * cfg.NUM_BATCH_FOR_MEM), embedding_dim=cfg.EMBEDDING_DIM)

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
					# ce_loss_weight = 1
     
					if itr >= cfg.CONTRAST_WARMUP:
						# Calculate triplet loss using boundary information

						if cfg.BDRY_LOSS:
							bdry_loss = bdry_fn(pred, feat, seg_label, sample['gt'].cuda())
							bdry_loss *= cfg.BDRY_LOSS_WEIGHT
							# ce_loss_weight -= cfg.BDRY_LOSS_WEIGHT
					else:
						bdry_loss = 0 * bdry_fn(pred, feat, seg_label, sample['gt'].cuda())

						# if cfg.SUPCON_LOSS:
						# 	# Supervised Contrastive Loss
						# 	# Calculate triplet loss when the memory bank is full with it's size.
						# 	# if cfg.ENABLE_MEMORY and (len(memory_bank.bank) == memory_bank.memory_size):
						# 	# 	supcon_loss = compute_triplet_loss(pred, feat, seg_label, gt_bdry, use_bank=True)
						# 	# else:
						# 	# 	supcon_loss = compute_triplet_loss(pred, feat, seg_label, gt_bdry, use_bank=False)

						# 	# if cfg.ENABLE_MEMORY:
						# 	# 	memory_bank.append(pred, feat, gt_bdry)
						# 	supcon_loss = supcon_fn(pred, feat, seg_label, cfg.NUM_ANCHORS)
       
						# 	supcon_loss *= cfg.SUPCON_LOSS_WEIGHT
						# 	ce_loss_weight -= cfg.SUPCON_LOSS_WEIGHT

					# loss = ce_loss * ce_loss_weight + bdry_loss
					loss = ce_loss + bdry_loss


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


