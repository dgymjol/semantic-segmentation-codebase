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
class deeplabv3_contrast(_deeplabv3):
	def __init__(self, cfg, **kwargs):
		super(deeplabv3_contrast, self).__init__(cfg, **kwargs)
  
		# for embeddings
		dim_in = self.backbone.OUTPUT_DIM
		proj_dim = cfg.EMBEDDING_DIM
  
		self.proj_head = nn.Sequential(nn.Conv2d(dim_in, dim_in, kernel_size=1),nn.BatchNorm2d(dim_in) ,nn.ReLU(), nn.Conv2d(dim_in, proj_dim, kernel_size=1))
  
        # org deeplabv3
		self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
		self.__initial__()

	def forward(self, x):
		n,c,h,w = x.size()
		x_bottom = self.backbone(x)[-1]
		feature = self.aspp(x_bottom)
		result = self.cls_conv(feature)
		result = F.interpolate(result,(h,w),mode='bilinear', align_corners=True)
		
		# embeddings
		embedding = F.normalize(self.proj_head(x_bottom), p=2, dim=1) # (B, 2048, 64, 64) -> (B, 256, 64, 64)
  
		return {'embed': embedding, 'pred_seg': result}

