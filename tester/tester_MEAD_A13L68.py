import sys
sys.path.append('..')

import torch
import argparse
import yaml
from dataloader.dataloader_import import *
from model.model_import import *
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='/root/TalkingHead/config/config_MEAD_A13L68.yaml')
args = parser.parse_args()

if __name__ == '__main__': 
    config = yaml.safe_load(open(args.config_file))
    dataloader = globals()[config['dataloader_name']](config, is_train=False)
    if torch.cuda.is_available():
        model = globals()[config['model_name']](config).cuda()
    
    #Load pretrain
    model.load(load_optim = True)
    model.eval()

    valid_running_loss = 0.0
    counter = 0
    for i, (x,y) in tqdm(enumerate(dataloader)):
        counter += 1

        y = y.reshape(*y.shape[0:2], -1)
        x, y = x.cuda(), y.cuda()
        pred = model(x)

        loss = model.calculate_loss(pred, y)
        valid_running_loss += loss.item()

    valid_epoch_loss = valid_running_loss / counter
    print(f'Val Loss: {valid_epoch_loss}')