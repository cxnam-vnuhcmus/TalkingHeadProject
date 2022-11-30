import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import *
from model import *

def get_dataset(dataset_name, dataset_path):
    dataset = globals()[dataset_name](dataset_path)
    return dataset

def get_dataloader(dataset, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    return dataloader

def get_model(model_name, **model_params):
    model = globals()[model_name](**model_params)
    if torch.cuda.is_available():
        model = model.cuda()
    return model

def train(model, dataloader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0
    for i, (x,y) in tqdm(enumerate(dataloader), f"Training epoch {epoch}: ", total=len(dataloader)):
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
        pred = model(x)
        
        optimizer.zero_grad()
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, epoch):
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for i, (x,y) in tqdm(enumerate(dataloader), f"Validate epoch {epoch}: ", total=len(dataloader)):
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            pred = model(x)

            loss = criterion(pred, y)        
            running_loss += loss.item()
    return running_loss / len(dataloader)

def save_model(model, epoch, optimizer=None, save_file='.'):
    dir_name = os.path.dirname(save_file)
    os.makedirs(dir_name, exist_ok=True)
    if optimizer is not None:
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, str(save_file))
    else:
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict()
        }, str(save_file))
        
def load_model(model, optimizer=None, save_file='.'):
    if next(model.parameters()).is_cuda and torch.cuda.is_available():
        checkpoint = torch.load(save_file, map_location=f'cuda:{torch.cuda.current_device()}')
    else:
        checkpoint = torch.load(save_file, map_location='cpu')
    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    epoch = checkpoint["epoch"]
    print(f"Load pretrained model at Epoch: {epoch}")
    return epoch

def save_plots(train_acc, val_acc, train_loss, val_loss, save_path='.'):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        val_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_path,'accuracy.png'))
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        val_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path,'loss.png'))
    