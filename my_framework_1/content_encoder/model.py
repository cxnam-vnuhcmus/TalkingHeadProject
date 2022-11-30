import torch.nn as nn
from torch.optim import *
from torch.nn import *
import numpy as np

class ContentEncoder(nn.Module):
    def __init__(self, **model_params):
        super(ContentEncoder, self).__init__()
        # define networks
        self.fc1 = nn.Sequential(
                nn.Linear(in_features=model_params['input_ndim'], out_features=512),
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
                nn.Linear(512, model_params['output_ndim']))

        self.init_model()
        self.num_params()

    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
    
    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print("Trainable Parameters: %.3fM" % parameters)
        return parameters
    
    #input shape (bs, no.frame, no.feature)
    def forward(self, x):
        bs, _ , ndim = x.shape
        x = x.view(-1, ndim)
        x = self.fc1(x)  
        x = x.view(bs, -1, 512)
        output, _ = self.LSTM(x)
        output = output.reshape(-1, 256)
        pred = self.fc2(output)
        pred = pred.view(bs, -1, pred.shape[1])
        return pred

    

