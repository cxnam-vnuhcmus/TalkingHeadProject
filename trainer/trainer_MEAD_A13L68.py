import sys
sys.path.append('..')

import torch
from torch import optim, nn
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
    optim = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criteon = nn.MSELoss()
    model.train()
    steps_per_epoch = np.ceil(len(dataloader.getDataset()) / config['batch_size']).astype(np.int32)
    for epoch in range(config['n_epoches']):
        for i, (x,y) in enumerate(dataloader):
            x = x.unsqueeze(0)
            y = y.reshape(y.shape[0], -1).unsqueeze(0)
            x, y = x.cuda(), y.cuda()
            pred = model(x)
            loss = criteon(pred, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            msg = f"| Epoch: {epoch}/{config['n_epoches']} ({i}/{steps_per_epoch}) | Loss: {loss:#.4} "
            sys.stdout.write("\r{%s}" % msg)
            sys.stdout.flush()
            