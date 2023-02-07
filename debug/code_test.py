import os
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from models.make_target_model import make_target_model
from mysql_queries import insertIntoTable
import mysql.connector
from mysql.connector import Error       # Catches exceptions that may occur during this process.
from inference import inference

class Config:
    pass
cfg = Config()
cfg.ori_shape = (256, 256)
cfg.image_crop_size = (224, 224)
cfg.normalize_mean = [0.5, 0.5, 0.5]
cfg.normalize_std = [0.5, 0.5, 0.5]
cfg.last_stride = 2
cfg.num_classes = 8
cfg.num_branches = cfg.num_classes + 1
cfg.backbone = 'resnet18' # 'resnet18', 'resnet50_ibn'
cfg.pretrained = "./weights/AffectNet_res18_acc0.6285.pth"
cfg.pretrained_choice = '' # '' or 'convert'
cfg.bnneck = True  
cfg.BiasInCls = False



os.system("cls")
img_path = './images2/test1.jpg'

transform = T.Compose([
        T.Resize(cfg.ori_shape),
        T.CenterCrop(cfg.image_crop_size),
        T.ToTensor(),
        T.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std),
    ])

# Build model and load pre-trained weights into it
print('Building model......')
model = make_target_model(cfg)
model.load_param(cfg)
print('Loaded pretrained model from {0}'.format(cfg.pretrained))

idx, prob = inference(model, img_path, transform)
print("inferred index: ", idx)
print("softmax: ", prob)
print("type of prob elements: ", type(prob[1]))
print(prob)




