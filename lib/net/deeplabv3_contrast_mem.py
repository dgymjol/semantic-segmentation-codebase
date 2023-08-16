# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from net.backbone import build_backbone
from net.operators import ASPP
from utils.registry import NETS

class _deeplabv3(nn.Module):	
	def __init__(self, cfg, batchnorm=nn.BatchNorm2d, **kwargs):
		super(_deeplabv3, self).__init__()
		self.cfg = cfg
		self.backbone = build_backbone(cfg.MODEL_BACKBONE, pretrained=cfg.MODEL_BACKBONE_PRETRAIN, **kwargs)
		self.batchnorm = batchnorm
		input_channel = self.backbone.OUTPUT_DIM	
    
		self.aspp = ASPP(dim_in=input_channel, 
				dim_out=cfg.MODEL_ASPP_OUTDIM, 
				rate=[0, 6, 12, 18],
				bn_mom = cfg.TRAIN_BN_MOM,
				has_global = cfg.MODEL_ASPP_HASGLOBAL,
				batchnorm = self.batchnorm)
	def __initial__(self):
		for m in self.modules():
			if m not in self.backbone.modules():
				if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
					nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				elif isinstance(m, self.batchnorm):
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)
		
	def forward(self, x):
		raise NotImplementedError

@NETS.register_module
class deeplabv3_contrast_mem(_deeplabv3):
	def __init__(self, cfg, **kwargs):
		super(deeplabv3_contrast_mem, self).__init__(cfg, **kwargs)
  
		# for embeddings
		dim_in = self.backbone.OUTPUT_DIM
		proj_dim = cfg.MODEL_PROJ_DIM
  
		self.proj_head = nn.Sequential(nn.Conv2d(dim_in, dim_in, kernel_size=1),nn.BatchNorm2d(dim_in) ,nn.ReLU(), nn.Conv2d(dim_in, proj_dim, kernel_size=1))
  
        # org deeplabv3
		self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
		self.__initial__()

		# for memory bank
		self.register_buffer("segment_queue", torch.randn(cfg.MODEL_NUM_CLASSES, cfg.MODEL_MEM_SIZE, proj_dim))
		self.segment_queue = nn.functional.normalize(self.segment_queue, p=2, dim=2)
		self.register_buffer("segment_queue_ptr", torch.zeros(cfg.MODEL_NUM_CLASSES, dtype=torch.long))

		self.register_buffer("pixel_queue", torch.randn(cfg.MODEL_NUM_CLASSES, cfg.MODEL_MEM_SIZE, proj_dim))
		self.pixel_queue = nn.functional.normalize(self.pixel_queue, p=2, dim=2)
		self.register_buffer("pixel_queue_ptr", torch.zeros(cfg.MODEL_NUM_CLASSES, dtype=torch.long))

	def forward(self, x):
		n,c,h,w = x.size()
		x_bottom = self.backbone(x)[-1]
		feature = self.aspp(x_bottom)
		result = self.cls_conv(feature)
  
  		# embeddings
		embedding = F.normalize(self.proj_head(x_bottom), p=2, dim=1) # (B, 2048, 64, 64) -> B, 256, 64, 64)
  
		return {'embed': embedding, 'seg': result, 'key': embedding.detach()}

