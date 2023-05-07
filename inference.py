import os
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from models.make_target_model import make_target_model
from mysql_queries import insertIntoTable
import mysql.connector
from mysql.connector import Error       # Catches exceptions that may occur during this process.
import yaml
with open('config\config.yml','r') as ymlConfigFile:
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


def inference(model, img_path, transform, printPredictions=False, printSoftmax=False, is_cuda=True):
    """Performs inference with the pytorch model. Inference is conducted for a single image.

    Args:
        model (_type_): a pytorch model
        img_path (string): filepath to the image to perform inference
        transform (_type_): 
        is_cuda (bool, optional): _description_. Defaults to True.

    Returns:
        int idx: the argmax of the softmax (i.e., the predicted emotion)
        inferenceDistribution: the softmax across the classes of emotions, rounded to 4 significant digits
    """
    img = Image.open(img_path).convert('RGB')    
    img_tensor = transform(img).unsqueeze(0)
    if is_cuda:
        img_tensor = img_tensor.cuda()
    
    model.eval()
    if is_cuda:
        model = model.cuda()

    pred = model(img_tensor)
    prob = F.softmax(pred, dim=-1)
    idx  = torch.argmax(prob.cpu()).item()
    inferenceDistribution = []

    key = {0: 'Neutral', 1:'Happy', 2:'Sad', 3:'Surprise', 4:'Fear', 5:'Disgust', 6:'Anger', 7:'Contempt'}
    if printPredictions == True:
        print('Predicted: {}'.format(key[idx]))
    if printSoftmax == True:
        print('Probabilities:')
    for i in range(cfg.num_classes):
        inferenceDistribution.append(round(prob[0, i].item(), 4))
        if printSoftmax == True:
            print('{} ----> {}'.format(key[i], inferenceDistribution[i]))
    return idx, inferenceDistribution

def multiplePredictions(model, img_path, transform, printFilenames=False):
    """
    Predict emotions of all images in the img_path, and stores it in the
    ferPredictions list. 
    
    Outputs:
        list ferPredictions: the class that is predicted via the argmax of the softmax
        list ferSoftmax: list of the distributions (floats) of each prediction
        list filenames: the filenames (string) associated with the other outputs
    """
    ferPrediction = []
    ferSoftmax = []
    filenames = []
    for filename in sorted(os.listdir(img_path)):
        idx, prob = inference(model, img_path+filename, transform)
        ferPrediction.append(idx)
        ferSoftmax.append(prob)
        filenames.append(filename)
        if printFilenames:
            print(filename)
    return ferPrediction, ferSoftmax, filenames

def predictionsForAffectnetValidation(model, img_filepath, transform):
    """Predict emotions for Affectnet validation set

    Args:
        model (_type_): pytorch model
        img_filepath (list): a list of filepaths (strings) to the images
        transform (_type_): 
        
    Outputs:
        predictions_list (list): list of integers for the predictions associated with the image file list
    """
    predictions_list = []
    for img in img_filepath:
        prediction, _ = inference(model, "D:/Projects/Datasets/AffectNet/Sample/affectNet_dataset_sample/"+img.split('../input/affectnetsample/')[-1], transform)
        predictions_list.append(prediction)
        
    return predictions_list
    

def predictionsForFer2013Validation(model, img_filepath, transform):
    """Predict emotions for fer2013 validation set

    Args:
        model (_type_): pytorch model
        img_filepath (list): a list of filepaths (strings) to the images
        transform (_type_): 
        
    Outputs:
        predictions_list (list): list of integers for the predictions associated with the image file list
    """
    predictions_list = []
    for img in img_filepath:
        prediction, _ = inference(model, "D:/Projects/Datasets/fer_2013/test/"+img, transform)
        predictions_list.append(prediction)
        
    return predictions_list




def main_inference(username, img_path):
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
    
    
    key = {0: 'Neutral', 1:'Happy', 2:'Sad', 3:'Surprise', 4:'Fear', 5:'Disgust', 6:'Anger', 7:'Contempt'}
    prediction, distribution = inference(model, img_path, transform)
    
      
    try:
        connection = mysql.connector.connect(
                                    host = config['mysql']['host'],
                                    database = config['mysql']['database'],
                                    user = config['mysql']['user'],
                                    password = config['mysql']['password'])
        
        if connection.is_connected():
            cursor = connection.cursor()
            print("Connected to mySQL database server. Cursor object created.")
        
        # Get the base filename in a directory
        filename = os.path.basename(img_path).split('/')[-1]
        
        print("{0} ---> {1}, \ndistribution: {2}".format(img_path, key[prediction], distribution))
        insertIntoTable(connection, cursor, username=username, prediction=key[prediction], valueArgmax=prediction, prob=distribution, filename=filename)    
            
    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("\nMySQL connection is closed")
