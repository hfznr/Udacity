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





#map labels
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

    
def arg_parser():
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('data_dir', nargs='*', action="store", default="./flowers")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=2)

    args = parser.parse_args()
    return args

    
image_datasets = []    
dataloaders = []
    
def load_data(train_dir,valid_dir,test_dir):
    
        # TODO: Define your transforms for the training, validation, and testing sets
    training_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    validation_data_transforms = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    testing_data_transforms = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])              


    # TODO: Load the datasets with ImageFolder
    
   
    train_data = datasets.ImageFolder(train_dir, transform=training_data_transforms) #0
    valid_data = datasets.ImageFolder(valid_dir, transform=validation_data_transforms)#1
    test_data = datasets.ImageFolder(test_dir, transform=testing_data_transforms)#2
    image_datasets.append(train_data)
    image_datasets.append(valid_dir)
    image_datasets.append(train_data)



    # TODO: Using the image datasets and the trainforms, define the dataloaders

    

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64,shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64,shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64,shuffle=True)

    dataloaders.append(trainloader)   
    dataloaders.append(validloader)
    dataloaders.append(testloader)

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
    
 
    
def train_model(model, trainloader, validloader, epochs, print_every,device,learning_rate,criterion,optimizer):
    #coppied from transfer learning example

    model.to(device)
    
    steps = 0
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:

            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:

                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    print("Train completed")
    
    return model

        
  

def test_network(model, testloader, device,criterion,optimizer):
    test_loss = 0
    accuracy = 0
    model.to(device)
    #inputs are images
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            log_ps = model.forward(inputs)
            temp_loss = criterion(log_ps, labels)

            test_loss += temp_loss.item()

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            result = 100 * accuracy/len(testloader)
           

    result = 100 * accuracy/len(testloader)      
    print(f"Result of test Accuracy : % {result}") 


def save_checkpoints(model, save_dir, train_data,criterion,optimizer,epochs,arch,hidden_units,dropout,lr):
    model.class_to_idx = train_data.class_to_idx
    
    checkpoint = {'structure' :arch,
                'hidden_units':hidden_units,
                'dropout':dropout,
                'learning_rate':lr,
                'no_of_epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx}
            
    """  
    checkpoint = {'structure': 'vgg16',
            'input_size': 25088,
            'dropout': 0.1,
            'output_size': 102,
            'learning_rate': 0.001,
            'classifier': model.classifier,
            'epochs': epochs,
            'optimizer': optimizer.state_dict(),
            'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx}"""
    
    torch.save(checkpoint,save_dir)
    
    
args = arg_parser()    
def main():

  
    arch = args.arch

    lr = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    gpu = args.gpu
    save_dir = args.save_dir
    data_dir = args.data_dir[0]

 
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

   

    load_data(train_dir,valid_dir,test_dir)
    train_data = image_datasets[0]
    valid_data = image_datasets[1]
    test_data  = image_datasets[2]

    trainloader = dataloaders[0]
    validloader = dataloaders[1]
    testloader = dataloaders[2]
    
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(arch,hidden_units,0.1)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    

    print_every = 5


    trained_model = train_model(model, trainloader, validloader, epochs, print_every,device,lr,criterion,optimizer)
                                            
    for p in model.parameters():
        p.requires_grad = False 

    test_network(trained_model, testloader, device,criterion,optimizer)

    save_checkpoints(trained_model, save_dir, train_data,criterion,optimizer,epochs,arch,hidden_units,0.1,lr)

if __name__ == '__main__': main()