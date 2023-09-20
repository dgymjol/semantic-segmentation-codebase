# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------
import torch
import argparse
import os
import sys
import cv2
import time

config_dict = {
		'EXP_NAME': 'deeplabv3_ddp_irn_bdcl_test2',
		'GPUS': 2,

		'DATA_NAME': 'VOCDataset',
		'DATA_YEAR': 2012,
		'DATA_AUG': True,
		'DATA_WORKERS': 2,
		'DATA_MEAN': [0.485, 0.456, 0.406],
		'DATA_STD': [0.229, 0.224, 0.225],
		'DATA_RESCALE': 512,
		'DATA_RANDOMSCALE': [0.5, 2.0],
		'DATA_RANDOM_H': 0,
		'DATA_RANDOM_S': 0,
		'DATA_RANDOM_V': 0,
		'DATA_RANDOMCROP': 512,
		'DATA_RANDOMROTATION': 0,
		'DATA_RANDOMFLIP': 0.5,
		'DATA_PSEUDO_GT': '/workspace/irn_pseudo_label',
		
		'MODEL_NAME': 'deeplabv3_contrast',
		'MODEL_BACKBONE': 'resnet101',
		'MODEL_BACKBONE_PRETRAIN': True,
		'MODEL_BACKBONE_DILATED': True,
		'MODEL_BACKBONE_MULTIGRID': False,
		'MODEL_BACKBONE_DEEPBASE': True,
		'MODEL_SHORTCUT_DIM': 48,
		'MODEL_OUTPUT_STRIDE': 8,
		'MODEL_ASPP_OUTDIM': 256,
		'MODEL_ASPP_HASGLOBAL': True,
		'MODEL_NUM_CLASSES': 21,
		'MODEL_FREEZEBN': False,

		'TRAIN_LR': 0.007,
		'TRAIN_MOMENTUM': 0.9,
		'TRAIN_WEIGHT_DECAY': 4e-5,
		'TRAIN_BN_MOM': 0.0003,
		'TRAIN_POWER': 0.9,
		'TRAIN_BATCHES': 16,
		'TRAIN_SHUFFLE': True,
		'TRAIN_MINEPOCH': 0,
		'TRAIN_ITERATION': 30000,
		'TRAIN_TBLOG': True,
  
		'CONTRAST_WARMUP' : 150, #3000
		'EMBEDDING_DIM' : 256,
  
		'SUPCON_LOSS' : False,
		'SUPCON_LOSS_WEIGHT' : 0.1, 
  
		'BDRY_LOSS' : True,
		'BDRY_LOSS_WEIGHT' : 0.1,
		'NUM_ANCHORS' : 200,
		'MEMORY_SIZE' : 5000,
		'TRIPLET_MARGIN' : 0.2, 
		'NEED_GT' : True,
  
		'TEMP' : 0.3,
		'EMBEDDING_DIN' : 256,
  
  
		'TEST_MULTISCALE': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
		'TEST_FLIP': True, 
		'TEST_CRF': False,
		'TEST_BATCHES': 1,		
}

config_dict['ROOT_DIR'] = '/workspace/semantic-segmentation-codebase'
config_dict['MODEL_SAVE_DIR'] = os.path.join(config_dict['ROOT_DIR'],'model',config_dict['EXP_NAME'])
config_dict['TRAIN_CKPT'] = None
config_dict['LOG_DIR'] = os.path.join(config_dict['ROOT_DIR'],'log',config_dict['EXP_NAME'])
config_dict['TEST_CKPT'] = os.path.join(config_dict['ROOT_DIR'],'model/deeplabv3_ddp_irn_bdcl_test2/deeplabv3_contrast_resnet101_VOCDataset_itr30000_all.pth')

sys.path.insert(0, os.path.join(config_dict['ROOT_DIR'], 'lib'))