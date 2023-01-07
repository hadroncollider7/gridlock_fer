import cv2
import os
import torch
from PIL import Image
from torchvision import transforms
from inference import inference
import torchvision.transforms as T
from models.make_target_model import make_target_model
from collections import Counter, deque
from mysql_queries import insertIntoTable
import mysql.connector
from mysql.connector import Error       # Catches exceptions that may occur during this process.
import yaml
with open('config.yml','r') as ymlConfigFile:
    config = yaml.safe_load(ymlConfigFile)


transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
key_mysql = {0: 'Neutral', 1:'Happy', 2:'Sad', 3:'Surprise', 4:'Fear', 5:'Disgust', 6:'Anger', 7:'Contempt'}
key = {0: '^(* . *)^ --> Neutral', 1:'<(^___^)> --> Happy', 2:'(TT____TT) --> Sad', 
       3:'(O___0) --> Surprise', 4:'\(>___<)/ --> Fear', 5:'Disgust', 6:'Angy!!! --> >:(', 7:'Contempt --> X______X'}


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


def load_img(path):
    img = Image.open(path)
    img = transformation(img).float()
    img = torch.autograd.Variable(img, requires_grad = True)
    img = img.unsqueeze(0)
    return img.to(device)


def computeModeOfList(myList):
    """Compute the mode of a list. Prioritizes the order of the element.

    Args:
        myList: A list or que.
    Returns:
        The most common element in the list.
    """
    count = Counter(myList)
    mostCommonElement = count.most_common(1)
    return mostCommonElement[0][0]

    
    
    
    
    
    

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
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
    
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # Initialize a que to store predictions
    predictionsList = []
    queSize = 7
    for i in range(queSize):
        predictionsList.append(0)
    predictionsList = deque(predictionsList)
    print("Initialized que of size {0}".format(len(predictionsList)))
    
    # Used for the numner of ticks until upload prediction to database server
    counterToUpload = 0
    uploadToDatabaseServer = False
    try:
        if uploadToDatabaseServer == True:
            connection = mysql.connector.connect(
                                        host = config['mysql']['host'],
                                        database = config['mysql']['database'],
                                        user = config['mysql']['user'],
                                        password = config['mysql']['password'])
            if connection.is_connected():
                cursor = connection.cursor()
                print("Connected to mySQL database server. Cursor object created.")
        
        capture = cv2.VideoCapture(0)
        while True:
            _, img = capture.read()
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Draw the rectangle around each face
            for (x, y, w, h) in faces:
                regionOfInterest = img[y:y+h, x:x+w]
                regionOfInterest = cv2.cvtColor(regionOfInterest, cv2.COLOR_BGR2GRAY)
                regionOfInterest = cv2.resize(regionOfInterest, (256,256))
                cv2.imwrite("regionOfInterest.jpg", regionOfInterest)
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            predictionsList.append(inference(model, 'regionOfInterest.jpg', transform))
            predictionsList.popleft()
            predictionsMode = computeModeOfList(predictionsList)
            print(predictionsList)
            print("length: {0}, mode: {1}".format(len(predictionsList), predictionsMode))
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            img = cv2.putText(img, key[predictionsMode], org, font, fontScale, color, thickness, cv2.LINE_AA)
            
            
            cv2.imshow('img', img)
            
            # Upload to database server after a certain number of ticks
            if uploadToDatabaseServer == True:
                noOfTicks = 5
                if counterToUpload % noOfTicks == noOfTicks - 1:
                    insertIntoTable(connection, cursor, 
                                    id=1, 
                                    name=key_mysql[predictionsMode], 
                                    value=predictionsMode, 
                                    filename='regionOfInterest.jpg')
                counterToUpload += 1
            
            # Stop if (Q) is pressed
            k = cv2.waitKey(30)
            if k==ord("q"):
                break
        
        # Release the VideoCapture object
        capture.release()

    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if uploadToDatabaseServer != False and connection.is_connected():
            cursor.close()
            connection.close()
            print("\nMySQL connection is closed")
    
