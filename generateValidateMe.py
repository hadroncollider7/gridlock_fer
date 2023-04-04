import os
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from models.make_target_model import make_target_model
from inference import multiplePredictions
import pandas as pd
import yaml
from evaluate_inference_model import sortTwoListsTogether

with open('public_config.yml', 'r') as ymlConfigFile:
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

def convert_predictions_to_string(predictions):
    """Converts the list of predictions into human readable strings.

    Args:
        predictions (list): List of integers representing the predictions. Affectnet key is used to interpret.

    Returns:
        list: The converted list of readable strings
    """
    # The AffectNet key
    key = {0: 'Neutral', 1:'Happy', 2:'Sad', 3:'Surprise', 4:'Fear', 5:'Disgust', 6:'Anger', 7:'Contempt', 8:'Ambiguous', 9:'Not a Face'}

    predictions_string = []
    for i in range(len(predictions)):
        predictions_string.append(key[predictions[i]])
    return predictions_string

if __name__ == "__main__":
    os.system("cls")

    # Load the prediction model
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

        
    # Read labels from the research group spreadsheet
    batch_no = config['evaluateInferenceModel']['batch_no']
    column = config['evaluateInferenceModel']['column']
    sheet = config['evaluateInferenceModel']['sheet']

  
    # Select batch to make inference
    if batch_no == 2:
        img_path = 'images/batch2/'
        spreadsheet = 'Batch2_Labels.xlsx'
    elif batch_no == 3:
        img_path = 'images/batch3/0-995/'
        spreadsheet = 'Batch3_Labels.xlsx'
    elif batch_no == 4:
        img_path = 'images/batch4/0-822/'
        spreadsheet = 'Batch4_Labels.xlsx'
    elif batch_no == 1:
        img_path = 'images/batch1/0-999/'
        spreadsheet = 'Batch1_Labels.xlsx'




    predictions, _, filenames = multiplePredictions(model, img_path, transform)
    predictions_string = convert_predictions_to_string(predictions)
    # Zip the lists and construct a dataframe with them
    data = pd.DataFrame(list(zip(filenames, predictions, predictions_string)),
                        columns=['filename', 'value', 'name'])

    # Write to excel file
    data.to_excel('validate_me_batch{0}.xlsx'.format(batch_no))
