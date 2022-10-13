import sys
sys.path.append('.')

import torch
import argparse
import yaml
from dataloader.dataloader_import import *
from model.model_import import *
import time

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='/root/TalkingHead/config/config_MEAD_A13L68.yaml')
args = parser.parse_args()

if __name__ == '__main__': 
    config = yaml.safe_load(open(args.config_file))
    dataloader = globals()[config['dataloader_name']](config, is_train=True)
    if torch.cuda.is_available():
        model = globals()[config['model_name']](config).cuda()
    model.train()

    steps_per_epoch = dataloader.getStepPerEpoch()

    #Load pretrain
    current_epoch = 0
    if config['use_pretrain'] == True:
        current_epoch = model.load(load_optim = True) + 1

    train_loss = []
    valid_loss = []
    for epoch in range(current_epoch, config['n_epoches']):
        train_running_loss = 0.0
        counter = 0
        for i, (x,y) in enumerate(dataloader):
            counter += 1
            # start_time = time.time() 

            # y = y.reshape(*y.shape[0:2], -1)
            x, y = x.cuda(), y.cuda()
            pred = model(x)

            loss = model.update(pred, y)
            train_running_loss += loss.item()

            #Log
            # step = model.get_step()
            msg = f"| Epoch: {epoch}/{config['n_epoches']} ({i}/{steps_per_epoch}) | Loss: {loss:#.4} | " \
            # f"{1./(time.time() - start_time):#.2} steps/s | Step: {step//1000}k | "
            sys.stdout.write("\r{%s}" % msg)
            sys.stdout.flush()

        train_epoch_loss = train_running_loss / counter
        train_loss.append(train_epoch_loss)
        #Save epoch model
        if epoch % config['save_epoch'] == 0:
            print(f"\nSave the model (epoch: {epoch})\n")
            model.save(epoch,train_epoch_loss, save_optim = True)

        #Save best model
        model.save_best_model(epoch,train_epoch_loss, save_optim = True)

    #Save plot
    model.save_plots([], [], train_loss, valid_loss)