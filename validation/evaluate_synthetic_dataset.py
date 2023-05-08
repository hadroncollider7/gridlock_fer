"""
Uses the synthetic dataset to validate the emotion prediction model.
Uses the predictionsForFer2013Validation() function. Change the filepath in the function in the inference module.
"""
# import module parent directory
import sys
sys.path.append('../gridlock_fer')

import os
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from models.make_target_model import make_target_model
from inference import predictionsForFer2013Validation
import pandas as pd
import yaml
import pickle
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from evaluate_affectnetData import save_list, read_list

with open('config\public_config.yml', 'r') as ymlConfigFile:
    config = yaml.safe_load(ymlConfigFile)

class Config:
    pass
cfg = Config()
cfg.ori_shape = (128, 128)
cfg.image_crop_size = (128, 128)
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

generate_predictions_list = False
os.system("cls")

transform = T.Compose([
        T.Resize(cfg.ori_shape),
        T.CenterCrop(cfg.image_crop_size),
        T.ToTensor(),
        T.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std),
    ])

key = {0: 'Neutral', 1:'Happy', 2:'Sad', 3:'Surprise', 4:'Fear', 5:'Disgust', 6:'Anger', 7:'Contempt'}

img_filenames_angry = os.listdir('D:/Projects/Datasets/synthetic_dataset/angry')
for i in range(len(img_filenames_angry)):
    img_filenames_angry[i] = "angry/"+img_filenames_angry[i]
groundLabels_angry = [6] * len(img_filenames_angry)

img_filenames_disgust = os.listdir('D:/Projects/Datasets/synthetic_dataset/disgust')
for i in range(len(img_filenames_disgust)):
    img_filenames_disgust[i] = "disgust/"+img_filenames_disgust[i]
groundLabels_disgust = [5] * len(img_filenames_disgust)

img_filenames_fear = os.listdir('D:/Projects/Datasets/synthetic_dataset/fear')
for i in range(len(img_filenames_fear)):
    img_filenames_fear[i] = "fear/"+img_filenames_fear[i]
groundLabels_fear = [4] * len(img_filenames_fear)

img_filenames_happy = os.listdir('D:/Projects/Datasets/synthetic_dataset/happy')
for i in range(len(img_filenames_happy)):
    img_filenames_happy[i] = "happy/"+img_filenames_happy[i]
groundLabels_happy = [1] * len(img_filenames_happy)

img_filenames_neutral = os.listdir('D:/Projects/Datasets/synthetic_dataset/neutral')
for i in range(len(img_filenames_neutral)):
    img_filenames_neutral[i] = "neutral/"+img_filenames_neutral[i]
groundLabels_neutral = [0] * len(img_filenames_neutral)

img_filenames_sad = os.listdir('D:/Projects/Datasets/synthetic_dataset/sad')
for i in range(len(img_filenames_sad)):
    img_filenames_sad[i] = "sad/"+img_filenames_sad[i]
groundLabels_sad = [2] * len(img_filenames_sad)

img_filenames_surprise = os.listdir('D:/Projects/Datasets/synthetic_dataset/surprise')
for i in range(len(img_filenames_surprise)):
    img_filenames_surprise[i] = "surprise/"+img_filenames_surprise[i]
groundLabels_surprise = [3] * len(img_filenames_surprise)

img_filenames_contempt = os.listdir('D:/Projects/Datasets/synthetic_dataset/contempt')
for i in range(len(img_filenames_contempt)):
    img_filenames_contempt[i] = "contempt/"+img_filenames_contempt[i]
groundLabels_contempt = [7] * len(img_filenames_contempt)


img_filenames = img_filenames_angry+img_filenames_disgust+img_filenames_fear+img_filenames_happy+img_filenames_neutral+img_filenames_sad+img_filenames_surprise+img_filenames_contempt
groundLabels = groundLabels_angry+groundLabels_disgust+groundLabels_fear+groundLabels_happy+groundLabels_neutral+groundLabels_sad+groundLabels_surprise+groundLabels_contempt

print(img_filenames[:5])

# Build model and load pre-trained weights into it
print('Building model......')
model = make_target_model(cfg)
model.load_param(cfg)
print('Loaded pretrained model from {0}'.format(cfg.pretrained))

if generate_predictions_list == True:
        predictions = predictionsForFer2013Validation(model, img_filenames, transform)
        save_list(predictions, 'validation/syntheticDataset_predictions.pkl')
        print("predictions object type: ", type(predictions))
        print("predictions element type: ", type(predictions[0]))
        print("predictions length", len(predictions))
else:
    predictions = read_list('validation/syntheticDataset_predictions.pkl')
    print("predictions object type: ", type(predictions))
    print("predictions element type: ", type(predictions[0]))
    print("predictions length", len(predictions))

print(len(predictions))
# Print the metrics
i = 0 

print(img_filenames[:20])
print(groundLabels[:20])
print(predictions[:20])
print(classification_report(groundLabels, predictions))
ConfusionMatrixDisplay.from_predictions(groundLabels, predictions)
plt.show()

