# Imports here
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
from PIL import Image 
from collections import OrderedDict
from os import listdir
import json
from torch import nn,optim
import torch.nn.functional as F
from torchvision import datasets, transforms,models
import argparse


def arg_parser():
    parser = argparse.ArgumentParser(description="predict.py")

    #parser.add_argument('input', default='./flowers/test/1/image_06752.jpg', nargs='?', action="store", type = str)
    #parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str)
    parser.add_argument('--dir', action="store",dest="data_dir", default="./flowers")
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")
    parser.add_argument('--image',type=str,help='impage file path ',required=True,dest = "image_path",default = './flowers/test/1/image_06752.jpg')
    parser.add_argument('--checkpoint',type=str,help='checkpoint file',required=True)
    parser.add_argument('--top_k',type=int,help=' top k matches .',default = 5)
    

    args = parser.parse_args()
    
    return args
def build_model(arch,hidden_value,dropout):
   
    input_size = 25088  
    output_size = 102
    
    if arch.lower() == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_value)),
            ('relu1', nn.ReLU()),
            ('dropout', nn.Dropout(dropout)),
            ('output', nn.Linear(hidden_value, output_size)),
            ('softmax', nn.LogSoftmax(dim=1))]))
    else:
        model = models.densenet121(pretrained=True)
        input_size =1024
        model.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_value)),
            ('relu1', nn.ReLU()),
            ('dropout', nn.Dropout(dropout)),
            ('output', nn.Linear(hidden_value, output_size)),
            ('softmax', nn.LogSoftmax(dim=1))]))
        

     
    return model      


def load_checkpoint(filepath='checkpoint.pth'):
    checkpoint = torch.load(filepath)
    lr=checkpoint['learning_rate']
    hidden_units = checkpoint['hidden_units']
    dropout = checkpoint['dropout']
    epochs = checkpoint['no_of_epochs']
    structure = checkpoint['structure']

    model = build_model(structure ,hidden_units, dropout)
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    for p in model.parameters():
        p.requires_grad = False 
       
    return model,optimizer
    """
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
    
    for p in model.parameters(): 
        p.requires_grad = False
        
    checkpoint = torch.load('checkpoint.pth')
    structure = checkpoint['structure']
    model.class_to_idx = checkpoint['class_to_idx']
    print(checkpoint['state_dict'])
    model.load_state_dict(checkpoint['state_dict'])
    for p in model.parameters():
        p.requires_grad = False
    return model
"""


def process_image(image):

    transformed = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image)

    transformed_image = transformed(img)
    return transformed_image

''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array

        '''
    
    
    # TODO: Process a PIL image for use in a PyTorch model
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(img_path , model, device, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    model.eval()
    
    img = process_image(img_path).numpy()
    img_last = torch.from_numpy(np.array([img])).float()

    with torch.no_grad():
        logps = model.forward(img_last.cuda())
        
    probability = torch.exp(logps).data
    pb = probability.topk(topk)
    frst = pb[0][0]
    second = pb[1][0]
    return [frst,second]
    
def sanity_checking(img_path,fst,scd,cat_to_name):
   
    image = process_image(img_path)


    i = 1
    
    labels = [cat_to_name[str(i+1)] for i in np.array(fst)]

    y = np.array(scd)
    
 
    
    for i in range (len(fst)):
        print("{} \t\t probability : {}".format(labels[i], y[i]))
        
    
    
    #f, axes = plt.subplots(figsize=(3,3))
    #axes1 = imshow(image, ax = plt)
    #axes1.axis('off')
    #axes1.title(cat_to_name[str(i)])


    #f,axes2 = plt.subplots(figsize=(3,3))
    #y_pos = np.arange(5)
    #axes2.set_yticks(y_pos)
    #axes2.set_yticklabels(labels)
    #axes2.invert_yaxis()
    #axes2.barh(y_pos, y)

    #plt.show()
    
args = arg_parser()
with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model,optimizer = load_checkpoint(args.checkpoint)
    criterion = nn.NLLLoss()
    model = model.to(device)
    img_path = args.image_path
    scd, fst = predict(img_path, model,device,args.top_k)
    
    sanity_checking(img_path,fst,scd,cat_to_name)
    
    print("!!!Finished Predicting!!!")
   
    

if __name__ == '__main__': main()