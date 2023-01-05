import os
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from models.make_target_model import make_target_model

from mysql_queries import insertIntoTable


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


def inference(model, img_path, transform, printPredictions=False, is_cuda=True):
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

    if printPredictions==True:
        key = {0: 'Neutral', 1:'Happy', 2:'Sad', 3:'Surprise', 4:'Fear', 5:'Disgust', 6:'Anger', 7:'Contempt'}
        print('Predicted: {}'.format(key[idx]))
        print('Probabilities:')
        for i in range(cfg.num_classes):
            print('{} ----> {}'.format(key[i], round(prob[0,i].item(), 4)))
    return idx

def multiplePredictions(model, img_path, transform, printFilenames=False):
    """
    Predict emotions of all images in the img_path, and stores it in the
    ferPredictions list. 
    
    Outputs:
        Two lists containing the predictiction string and the filename, respectively.
    """
    ferPrediction = []
    filenames = []
    for filename in sorted(os.listdir(img_path)):
        ferPrediction.append(inference(model, img_path+filename, transform, is_cuda=True))
        filenames.append(filename)
        if printFilenames:
            print(filename)
    return ferPrediction, filenames
    




if __name__ == '__main__':
    os.system("cls")
    img_path = './images/'

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
    predictions, filenames = multiplePredictions(model, img_path, transform)
    for i in range(len(predictions)):
        print("{0} ---> {1}".format(filenames[i], key[predictions[i]]))
    for i in range(len(predictions)):
        insertIntoTable(id=i+1, name=key[predictions[i]], value=predictions[i], filename=filenames[i])    
    