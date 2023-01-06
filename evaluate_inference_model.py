"""
Evaluate the accuracey of the inference model based the Tang Research Group's dataset.
"""

import os
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from models.make_target_model import make_target_model
from mysql_queries import insertIntoTable

from inference import inference, multiplePredictions

import pandas as pd


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



if __name__ == '__main__':
    os.system("cls")
    img_path = 'images/batch4/0-822/'

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

    # The AffectNet key
    key = {0: 'Neutral', 1:'Happy', 2:'Sad', 3:'Surprise', 4:'Fear', 5:'Disgust', 6:'Anger', 7:'Contempt', 8:'Amiguous', 9:'Not a Face'}
    # Convert the research group key to the AffectNet key
    key_researchGroup = {0:0, 1:6, 2:7, 3:5, 4:4, 5:1, 6:2, 7:3, 8:8, 9:9}
    
    # Read labels from the research group spreadsheet
    spreadsheet = 'Batch4_Labels.xlsx'
    column = 'chau'
    sheet = 'Labels'
    dataframe1 = pd.read_excel(spreadsheet, sheet_name=[sheet], usecols=['Image', column])
    labelFilenames = list(dataframe1[sheet]['Image'])
    subjectiveLabels = list(dataframe1[sheet][column])
    
    # Remove bad images
    notAFace_images = []
    for i in range(len(labelFilenames)):
        if (subjectiveLabels[i] == 9) or (subjectiveLabels[i] == 8):
            notAFace_images.append(labelFilenames[i])
    
    # Convert type float to type int
    for i in range(len(subjectiveLabels)):
        subjectiveLabels[i] = int(subjectiveLabels[i])
        
    # Create dictionary with filename and label
    labelDictionary = {}
    for i in range(len(labelFilenames)):
        labelDictionary[labelFilenames[i]] = subjectiveLabels[i]
    
    predictions, filenames = multiplePredictions(model, img_path, transform)
    count = 0
    accuracy_score = 0
    for i in range(len(predictions)): 
        AFace = True
        for j in range(len(notAFace_images)):
            if filenames[i] == notAFace_images[j]:
                AFace = False
        if AFace:
            print("{3}: {0}, predicted: {1}, label: {2}".format(filenames[i], key[predictions[i]], key[key_researchGroup[labelDictionary[filenames[i]]]], count))
            if  key[predictions[i]] == key[key_researchGroup[labelDictionary[filenames[i]]]]: 
                accuracy_score += 1
            count += 1 
    
    print('Total predictions: ', len(predictions))
    print('number of bad images: ', len(notAFace_images))
    print('No. of matches: ', accuracy_score)
    print('Accuracy: ', accuracy_score/(len(predictions) - len(notAFace_images)))

