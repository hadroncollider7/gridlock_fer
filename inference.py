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
with open('config.yml','r') as ymlConfigFile:
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


def inference(model, img_path, transform, printSoftmax=False, is_cuda=True):
    """Performs inference with the pytorch model

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
        list ferPredictions
        list ferSoftmax: list of the distributions of each prediction
        list filenames
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
    




if __name__ == '__main__':
    os.system("cls")
    img_path = config['sourcePaths']['imagePath']

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
    predictions, distributions, filenames = multiplePredictions(model, img_path, transform)
    
    try:
        connection = mysql.connector.connect(
                                    host = config['mysql']['host'],
                                    database = config['mysql']['database'],
                                    user = config['mysql']['user'],
                                    password = config['mysql']['password'])
        
        if connection.is_connected():
            cursor = connection.cursor()
            print("Connected to mySQL database server. Cursor object created.")
            
        for i in range(len(predictions)):
            print("{0} ---> {1}, \ndistribution: {2}".format(filenames[i], key[predictions[i]], distributions[i]))
            insertIntoTable(connection, cursor, id=i+1, name=key[predictions[i]], value=predictions[i], filename=filenames[i])    
            
    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("\nMySQL connection is closed")
    
