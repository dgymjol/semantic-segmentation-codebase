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
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from net.sync_batchnorm.replicate import patch_replication_callback
from utils.configuration import Configuration
from utils.finalprocess import writelog
from utils.imutils import img_denorm
from net.sync_batchnorm import SynchronizedBatchNorm2d
from utils.visualization import generate_vis, max_norm
from tqdm import tqdm

class PixelContrastLoss(nn.Module):
    def __init__(self):
        super(PixelContrastLoss, self).__init__()

        self.temperature = cfg.TEMP
        self.base_temperature = cfg.BASE_TEMP

        self.ignore_label = 255

        self.max_samples = cfg.MAX_SAMPLES
        self.max_views = cfg.MAX_VIEWS

    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _sample_negative(self, Q):
        class_num, cache_size, feat_size = Q.shape

        X_ = torch.zeros((class_num * cache_size, feat_size)).float().cuda()
        y_ = torch.zeros((class_num * cache_size, 1)).float().cuda()
        sample_ptr = 0
        for ii in range(class_num):
            if ii == 255: continue
            this_q = Q[ii, :cache_size, :]

            X_[sample_ptr:sample_ptr + cache_size, ...] = this_q
            y_[sample_ptr:sample_ptr + cache_size, ...] = ii
            sample_ptr += cache_size

        return X_, y_

    def _contrastive(self, X_anchor, y_anchor, queue=None):
        anchor_num, n_view = X_anchor.shape[0], X_anchor.shape[1]

        y_anchor = y_anchor.contiguous().view(-1, 1)
        anchor_count = n_view
        anchor_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        if queue is not None:
            X_contrast, y_contrast = self._sample_negative(queue)
            y_contrast = y_contrast.contiguous().view(-1, 1)
            contrast_count = 1
            contrast_feature = X_contrast
        else:
            y_contrast = y_anchor
            contrast_count = n_view
            contrast_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        mask = torch.eq(y_anchor, y_contrast.T).float().cuda()

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)

        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None, queue=None):
        # (Pdb) labels.shape
		# torch.Size([16, 512, 512])
		# (Pdb) predict.shape
		# torch.Size([16, 64, 64])
		# (Pdb) feats.shape
		# torch.Size([16, 256, 64, 64])
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest') #torch.Size([16, 1, 64, 64])
        labels = labels.squeeze(1).long() #torch.Size([16, 64, 64])
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1) #torch.Size([16, 4096])
        predict = predict.contiguous().view(batch_size, -1) #torch.Size([16, 4096])
        feats = feats.permute(0, 2, 3, 1) #torch.Size([16, 64, 64, 256])
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1]) # torch.Size([16, 4096, 256])

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        loss = self._contrastive(feats_, labels_, queue)
        return loss


class ContrastCELoss(nn.Module):
    def __init__(self):
        super(ContrastCELoss, self).__init__()

        ignore_index = 255

        self.seg_criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.contrast_criterion = PixelContrastLoss()

    def forward(self, preds, target, with_embed=False):
        h, w = target.size(1), target.size(2)

        assert "seg" in preds
        assert "embed" in preds

        seg = preds['seg']
        embedding = preds['embed']

        pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, target)

        segment_queue = preds['segment_queue']
        pixel_queue = preds['pixel_queue']
        queue = torch.cat((segment_queue, pixel_queue), dim=1)
        
        _, predict = torch.max(seg, 1)
        loss_contrast = self.contrast_criterion(embedding, target, predict, queue)

        if with_embed is True:
            return loss + cfg.LOSS_WEIGHT * loss_contrast

        return loss + 0 * loss_contrast  # just a trick to avoid errors in distributed training

def dequeue_and_enqueue(keys, labels, segment_queue, segment_queue_ptr, pixel_queue, pixel_queue_ptr):
    
        batch_size = keys.shape[0]
        feat_dim = keys.shape[1]

        labels = labels[:, ::cfg.NETWORK_STRIDE, ::cfg.NETWORK_STRIDE]

        for bs in range(batch_size):
            this_feat = keys[bs].contiguous().view(feat_dim, -1)
            this_label = labels[bs].contiguous().view(-1)
            this_label_ids = torch.unique(this_label)
            this_label_ids = [x for x in this_label_ids if x < 255]

            for lb in this_label_ids:
                idxs = (this_label == lb).nonzero()

                # segment enqueue and dequeue
                feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1)
                ptr = int(segment_queue_ptr[lb])
                segment_queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
                segment_queue_ptr[lb] = (segment_queue_ptr[lb] + 1) % cfg.MODEL_MEM_SIZE

                # pixel enqueue and dequeue
                num_pixel = idxs.shape[0]
                perm = torch.randperm(num_pixel)
                K = min(num_pixel, cfg.PIXEL_UPDATE_FREQ)
                idxs_perm = idxs[perm]
                feat = this_feat[:,idxs_perm[:K]].squeeze()
                if len(feat.size()) == 1:
                    feat = feat.unsqueeze(dim=1)
                feat = torch.transpose(feat, 0, 1)
                ptr = int(pixel_queue_ptr[lb])

                if ptr + K >= cfg.MODEL_MEM_SIZE:
                    pixel_queue[lb, -K:, :] = nn.functional.normalize(feat, p=2, dim=1)
                    pixel_queue_ptr[lb] = 0
                else:
                    pixel_queue[lb, ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
                    pixel_queue_ptr[lb] = (pixel_queue_ptr[lb] + K) % cfg.MODEL_MEM_SIZE

cfg = Configuration(config_dict)
def train_net():
	period = 'train'
	transform = 'weak'
	dataset = generate_dataset(cfg, period=period, transform=transform)
	def worker_init_fn(worker_id):
		np.random.seed(1 + worker_id)
	dataloader = DataLoader(dataset, 
				batch_size=cfg.TRAIN_BATCHES, 
				shuffle=cfg.TRAIN_SHUFFLE, 
				num_workers=cfg.DATA_WORKERS,
				pin_memory=True,
				drop_last=True,
				worker_init_fn=worker_init_fn)
	
	if cfg.GPUS > 1:
		net = generate_net(cfg, batchnorm=SynchronizedBatchNorm2d, dilated=cfg.MODEL_BACKBONE_DILATED, multi_grid=cfg.MODEL_BACKBONE_MULTIGRID, deep_base=cfg.MODEL_BACKBONE_DEEPBASE)
	else:
		net = generate_net(cfg, batchnorm=nn.BatchNorm2d, dilated=cfg.MODEL_BACKBONE_DILATED, multi_grid=cfg.MODEL_BACKBONE_MULTIGRID, deep_base=cfg.MODEL_BACKBONE_DEEPBASE)
	if cfg.TRAIN_CKPT:
		net.load_state_dict(torch.load(cfg.TRAIN_CKPT),strict=True)
		print('load pretrained model')
	if cfg.TRAIN_TBLOG:
		from tensorboardX import SummaryWriter
		# Set the Tensorboard logger
		tblogger = SummaryWriter(cfg.LOG_DIR)	

	print('Use %d GPU'%cfg.GPUS)
	device = torch.device(0)
	if cfg.GPUS > 1:
		net = nn.DataParallel(net)
		patch_replication_callback(net)
		parameter_source = net.module
	else:
		parameter_source = net
	net.to(device)		
 
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
	tblogger = SummaryWriter(cfg.LOG_DIR)
	criterion = ContrastCELoss()
	scaler = torch.cuda.amp.GradScaler()
	with tqdm(total=max_itr) as pbar:
		for epoch in range(cfg.TRAIN_MINEPOCH, max_epoch):
			for i_batch, sample in enumerate(dataloader):
       
				with_embed = True if itr >= cfg.CONTRAST_WARMUP else False
                
				now_lr = adjust_lr(optimizer, itr, max_itr, cfg.TRAIN_LR, cfg.TRAIN_POWER)
				optimizer.zero_grad()

				inputs, seg_label = sample['image'], sample['segmentation']
				n,c,h,w = inputs.size()

				with torch.cuda.amp.autocast():
					outputs = net(inputs.to(0))
     
					outputs['pixel_queue'] = net.module.pixel_queue
					outputs['pixel_queue_ptr'] = net.module.pixel_queue_ptr
					outputs['segment_queue'] = net.module.segment_queue
					outputs['segment_queue_ptr'] = net.module.segment_queue_ptr    
     
     
					loss = criterion(outputs, seg_label.to(0), with_embed)
     
					dequeue_and_enqueue(keys=outputs['key'], labels=seg_label.to(0),
                                          segment_queue=net.module.segment_queue,
                                          segment_queue_ptr=net.module.segment_queue_ptr,
                                          pixel_queue=net.module.pixel_queue,
                                          pixel_queue_ptr=net.module.pixel_queue_ptr)
     
				scaler.scale(loss).backward()
				scaler.step(optimizer)
				scaler.update()

				pbar.set_description("loss=%g " % (loss.item()))
				pbar.update(1)
				time.sleep(0.001)
				#print('epoch:%d/%d\tbatch:%d/%d\titr:%d\tlr:%g\tloss:%g' % 
				#	(epoch, max_epoch, i_batch, len(dataset)//(cfg.TRAIN_BATCHES),
				#	itr+1, now_lr, loss.item()))
				if cfg.TRAIN_TBLOG and itr%100 == 0:
					inputs1 = img_denorm(inputs[-1].cpu().numpy()).astype(np.uint8)
					label1 = sample['segmentation'][-1].cpu().numpy()
					label_color1 = dataset.label2colormap(label1).transpose((2,0,1))

					n,c,h,w = inputs.size()
					# seg_vis1 = torch.argmax(pred1[-1], dim=0).detach().cpu().numpy()
					# seg_color1 = dataset.label2colormap(seg_vis1).transpose((2,0,1))

					tblogger.add_scalar('loss', loss.item(), itr)
					tblogger.add_scalar('lr', now_lr, itr)
					tblogger.add_image('Input', inputs1, itr)
					tblogger.add_image('Label', label_color1, itr)
					# tblogger.add_image('SEG1', seg_color1, itr)
				itr += 1
				if itr>=max_itr:
					break
			save_path = os.path.join(cfg.MODEL_SAVE_DIR,'%s_%s_%s_epoch%d.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,epoch))
			torch.save(parameter_source.state_dict(), save_path)
			print('%s has been saved'%save_path)
			remove_path = os.path.join(cfg.MODEL_SAVE_DIR,'%s_%s_%s_epoch%d.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,epoch-1))
			if os.path.exists(remove_path):
				os.remove(remove_path)
			
	save_path = os.path.join(cfg.MODEL_SAVE_DIR,'%s_%s_%s_itr%d_all.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,cfg.TRAIN_ITERATION))
	torch.save(parameter_source.state_dict(),save_path)
	if cfg.TRAIN_TBLOG:
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


