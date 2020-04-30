import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms, models
import pandas as pd

import argparse
from PIL import Image
import json


def get_train_args():
    parser = argparse.ArgumentParser(description='Provide a command line argument.')

    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('--dir', type=str, default='flowers/', help="path to the folder of the pet images")
    parser.add_argument('--arch', choices=['vgg16', 'densenet121'], default='vgg16', help="model architechture")
    parser.add_argument('--lr', type=str, default='0.001', help="learn rate")
    parser.add_argument('--hid', type=int, default='4096', help="Hidden units")
    parser.add_argument('--epo', type=int, default='1', help="Epochs")
    parser.add_argument('--dev', type=str, default='gpu', help="Device")

    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()

def get_predict_args():
    parser = argparse.ArgumentParser(description='Provide a command line argument.')

    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('--img', type=str, default="flowers/test/1/image_06752.jpg", help="path to the image")
    parser.add_argument('--json', type=str, default='cat_to_name.json', help="provide a json file")
    parser.add_argument('--dev', type=str, default='gpu', help="Device")
    parser.add_argument('--check', type=str, default='checkpoint.pth', help="Checkpoint Path")

    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()


def save_checkpoint(model, image_datasets, architecture="vgg16", hidden_layers="4096"):
    model.class_to_idx = image_datasets['train'].class_to_idx

    try:
        hidden_layers = int(hidden_layers)
    except:
        return "hidden layers is not a number"
    
    if architecture == "vgg16":
        checkpoint = {'input_size': 25088,
                      'output_size': 102,
                      "model": "vgg16",
                      'hidden_layers': hidden_layers,
                      'state_dict': model.state_dict(),
                      'class_to_idx':model.class_to_idx}
        
    elif architecture == "densenet121":
        checkpoint = {'input_size': 1024,
                      'output_size': 102,
                      "model": "densenet121",
                      'hidden_layers': hidden_layers,
                      'state_dict': model.state_dict(),
                      'class_to_idx':model.class_to_idx}
    else:
        return "you have to provide vgg16 or densenet121"

    torch.save(checkpoint, 'checkpoint.pth')


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if checkpoint["model"] == "vgg16":
        model = models.vgg16(pretrained=True)
    elif checkpoint["model"] == "densenet121":
        model = models.densenet121(pretrained=True)
    else:
        return "You did not provide a vgg16 or densenet121"
    
    for param in model.parameters():
        param.requires_grad = False

    
    classifier = nn.Sequential(nn.Linear(checkpoint["input_size"],checkpoint["hidden_layers"]),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(checkpoint["hidden_layers"],checkpoint["hidden_layers"]),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(checkpoint["hidden_layers"],102),
                               nn.LogSoftmax(dim=1))

    model.classifier = classifier
    model.class_to_idx = checkpoint["class_to_idx"]
    model.load_state_dict(checkpoint['state_dict'])
    
    model.to(device);
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    tran = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    image = tran(Image.open(image))

    return image

def create_labels(labels, file):
    
    with open(file, 'r') as f:
        categories = json.load(f)
        
    result = []
    for label in labels:
        label = label + 1
        result.append(categories[str(label)])
        
    return result


