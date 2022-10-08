import torch
import torch.nn as nn
import numpy as np
import os
from model.model_talkinghead import Model_TalkingHead

class Model_LSTM(Model_TalkingHead):
    def __init__(self, config):
        super(Model_LSTM, self).__init__()
        self.config = config
        # define networks
        self.fc1 = nn.Sequential(
                nn.Linear(in_features=config['input_ndim'], out_features=512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 512),
                )
        self.LSTM = nn.LSTM(input_size=512,
                            hidden_size=256,
                            num_layers=3,
                            dropout=0,
                            bidirectional=False,
                            batch_first=True)
        self.fc2 = nn.Sequential(
                nn.Linear(in_features=256, out_features=512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, config['output_ndim']))

        self.init_model()
        self.num_params()
        self.register_buffer("step", torch.zeros(1, dtype=torch.long))

    #input shape (bs, no.frame, no.feature)
    def forward(self, x):
        self.step += 1
        bs, _ , ndim = x.shape
        x = x.view(-1, ndim)
        x = self.fc1(x)  
        x = x.view(bs, -1, 512)
        output, _ = self.LSTM(x)
        output = output.reshape(-1, 256)
        pred = self.fc2(output)
        pred = pred.view(bs, -1, pred.shape[1])
        return pred

    

