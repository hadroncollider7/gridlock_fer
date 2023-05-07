"""
Uses the Affectnet dataset to generate a list of predictions for validation. The list of predictions
is saved to disk via pickle.dump()
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
from inference import predictionsForAffectnetValidation
import pandas as pd
import yaml
import pickle
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

with open('config\public_config.yml', 'r') as ymlConfigFile:
    config = yaml.safe_load(ymlConfigFile)

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

def save_list(mylist, filename):
    """Saves a list to the disk.

    Args:
        mylist (list)
        filename (string)
    """
    open_file = open(filename, "wb")
    pickle.dump(mylist, open_file)
    open_file.close()
    
def read_list(filename):
    """Loads a list from the disk.

    Args:
        filename (string)

    Returns:
        list
    """
    open_file = open(filename, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()
    return loaded_list

if __name__ == "__main__":
    generate_predictions_list = False
    os.system("cls")

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
    key = {0: 'Neutral', 1:'Happy', 2:'Sad', 3:'Surprise', 4:'Fear', 5:'Disgust', 6:'Anger', 7:'Contempt', 8:'Ambiguous', 9:'Not a Face'}

    label_path = config['affectnet_dataset']['label_path']
    print('label path: ', label_path)

    # read csv file containing ground labels (int from 1 to 8) and filepath to the respective image (string)
    labels_df = pd.read_csv(label_path, usecols=['emotion', 'image'])

    if generate_predictions_list == True:
        # print(list(labels_df.loc[:5]['image'])[0].split('../input/affectnetsample/')[-1])
        predictions = predictionsForAffectnetValidation(model, list(labels_df.loc[:]['image']), transform)
        save_list(predictions, 'validation/affectnet_valSet_predictions.pkl')
        print("predictions object type: ", type(predictions))
        print("predictions element type: ", type(predictions[0]))
        print("predictions length", len(predictions))
    else:
        predictions = read_list('validation/affectnet_valSet_predictions.pkl')
        print("predictions object type: ", type(predictions))
        print("predictions element type: ", type(predictions[0]))
        print("predictions length", len(predictions))
        
    # Print the metrics
    ground_labels = list(labels_df.loc[:]['emotion'])
    i = 0 
    for label in ground_labels:
        ground_labels[i] = int(label - 1)
        i += 1
        
    print(ground_labels[:20])
    print(predictions[:20])
    print(classification_report(ground_labels, predictions))
    ConfusionMatrixDisplay.from_predictions(ground_labels, predictions)
    plt.show()

