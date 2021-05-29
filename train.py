import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import argparse
import time
from models import *

#parse arguments
parser = argparse.ArgumentParser(description='Mobilenet v3 training')
parser.add_argument('--model', type=str, default='small', help='large or small (small by default)')
parser.add_argument('--width', type=float, default=1.0, help='width multiplier (1.0 by default)')
parser.add_argument('--iter', type=int, default=20, help='number of iterations to train (20 by default)')
parser.add_argument('--batch', type=int, default=128, help='batch size (128 by default)')
args = parser.parse_args()

#define transformations
CIFAR100_MEANS = (0.5071, 0.4865, 0.4409) #precomputed channel means of CIFAR100(train) for normalization
CIFAR100_STDS = (0.2673, 0.2564, 0.2762) #precomputed standard deviations
transformations = {
    'train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEANS, CIFAR100_STDS)
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEANS, CIFAR100_STDS)
    ])
}

#load datasets and define loaders
data_path = 'cifar100'
cifar100 = datasets.CIFAR100(data_path, train=True, download=True, transform=transformations['train'])
cifar100_val = datasets.CIFAR100(data_path, train=False, download=True, transform=transformations['val'])
train_loader = torch.utils.data.DataLoader(cifar100, batch_size=args.batch, shuffle=True, pin_memory=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(cifar100_val, batch_size=args.batch, shuffle=False, pin_memory=True, drop_last=True)

#choose model to train
if args.model == 'large':
    model = Mobilenet_v3_large(args.width)
else:
    model = Mobilenet_v3_small(args.width)

#number of iterations
n_epochs = args.iter

#optimizer and scheduler
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 20, 30], gamma=0.1)

#loss function (standard cross-entropy taking logits as inputs)
loss_fn = nn.CrossEntropyLoss()

#train on GPU if CUDA is available, else on CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#print training information
print("")
if torch.cuda.is_available():
    hardware = "GPU " + str(device) 
else:
    hardware = "CPU (CUDA was not found)" 
print("Training information:")
print("model:", model.name())
print("hardware:", hardware)
print("iterations:", n_epochs)
print("width multiplier:", args.width)
print("batch size:", args.batch)
print("")

#training loop
lowest_val_loss = np.inf #used for saving the best model with lowest validation loss
for epoch in range(1, n_epochs+1):
    #train model
    start_time = time.time()
    train_losses = []
    model.train()
    for i, (imgs, labels) in enumerate(train_loader):
        print("train batch " + str(i+1) + "/" + str(len(train_loader)), end="\r", flush=True) 
        imgs, labels = imgs.to(device), labels.to(device)
        batch_size = imgs.shape[0]
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    scheduler.step()
    
    print("                         ", end="\r", flush=True) #delete output from train counter to not interfere with validation counter (probably can be done better)

    #validate model
    with torch.no_grad():
        model.eval()
        correct_labels = 0
        all_labels = 0
        val_losses = []
        for i, (imgs, labels) in enumerate(val_loader):
            print("valid batch " + str(i+1) + "/" + str(len(val_loader)), end="\r", flush=True) 
            imgs, labels = imgs.to(device), labels.to(device)
            batch_size = imgs.shape[0]
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            val_losses.append(loss.item())
            _, preds = torch.max(outputs, dim=1) #predictions
            matched = preds == labels #comparison with ground truth
            correct_labels += float(torch.sum(matched))
            all_labels += float(batch_size)

        val_accuracy = correct_labels / all_labels #compute top-1 accuracy on validation data 
    
    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    #save best model so far
    if val_loss < lowest_val_loss:
        lowest_val_loss = val_loss
        torch.save(model.state_dict(), './trained_models/' + model.name() + '_best' + '.pth')
    
    end_time = time.time()
    
    #print iteration results
    print("Epoch: %d/%d, lr: %f, train_loss: %f, val_loss: %f, val_acc: %f, time(sec): %f" % (epoch, n_epochs, optimizer.param_groups[0]['lr'], train_loss, val_loss, val_accuracy, end_time - start_time))
