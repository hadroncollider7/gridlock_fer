import cv2
import os
import torch
from PIL import Image
from torchvision import transforms
from inference import inference
import torchvision.transforms as T
from models.make_target_model import make_target_model

transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
key = {0: '^(* . *)^', 1:'<(^___^)>', 2:'(TT____TT)', 3:'(O___0)', 4:'\(>___<)/', 5:'Disgust', 6:'Angy!!!', 7:'Contempt'}


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
    face_cascade = cv2. CascadeClassifier('haarcascade_frontalface_default.xml')
    
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
            
        prediction = inference(model, 'regionOfInterest.jpg', transform)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        img = cv2.putText(img, key[prediction], org, font, fontScale, color, thickness, cv2.LINE_AA)
        
        
        cv2.imshow('img', img)
        # Stop if (Q) is pressed
        k = cv2.waitKey(30)
        if k==ord("q"):
            break
    
    
    # Release the VideoCapture object
    capture.release()
        
        