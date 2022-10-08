import sys
sys.path.append('..')

import torch
import argparse
import yaml
import numpy as np
from dataloader.dataloader_import import *
from model.model_import import *

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='/root/TalkingHead/config/config_MEAD_A13L68.yaml')
args = parser.parse_args()

if __name__ == '__main__': 
    config = yaml.safe_load(open(args.config_file))
    dataloader = globals()[config['dataloader_name']](config)
    if torch.cuda.is_available():
        model = globals()[config['model_name']](config).cuda()
    model.train()

    steps_per_epoch = dataloader.getStepPerEpoch()

    for epoch in range(config['n_epoches']):
        for i, (x,y) in enumerate(dataloader):
            x = x.unsqueeze(0)
            y = y.reshape(y.shape[0], -1).unsqueeze(0)
            x, y = x.cuda(), y.cuda()
            pred = model(x)

            model.update(pred, y)

            msg = f"| Epoch: {epoch}/{config['n_epoches']} ({i}/{steps_per_epoch}) | Loss: {1.0:#.4} "
            sys.stdout.write("\r{%s}" % msg)
            sys.stdout.flush()
            