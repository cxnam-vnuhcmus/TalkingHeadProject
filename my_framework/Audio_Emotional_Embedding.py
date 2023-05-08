import json
import argparse
import numpy as np
import sys
import os
import random
import datetime
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from modules.util_module import *
from modules.net_module import conv2d
from modules.face_visual_module import connect_face_keypoints
from evaluation.evaluation_landmark import *

filename = os.path.basename(__file__).split('.')[0]

dataset = 'MEAD'
parser = argparse.ArgumentParser()
parser.add_argument('--train_dataset_path', type=str, default=f'data/train_emo_{dataset}.json')
parser.add_argument('--val_dataset_path', type=str, default=f'data/val_emo_{dataset}.json')

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=1.0e-4)
parser.add_argument('--n_epoches', type=int, default=500)

parser.add_argument('--save_path', type=str, default=f'result_{filename}_{dataset}')
parser.add_argument('--use_pretrain', action='store_true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--val', action='store_true')
args = parser.parse_args()

class FaceDataset(Dataset):
    def __init__(self, path=None):
        self.data_path = []
        if path is not None:
            with open(path, 'r') as f:
                self.data_path = json.load(f)
        
        torch.autograd.set_detect_anomaly(True)
        
    def __getitem__(self, index):
        #/root/Datasets/Features/M030/aufeat25/contempt/level_1/00015/00017.npy
        data_path = self.data_path[index]  
        parts = data_path.split('/')
        emo_mapping = ['angry', 'disgusted', 'contempt', 'fear', 'happy', 'sad', 'surprised', 'neutral']
        lv_mapping = ['level_1','level_2','level_3']
        emo_embed = emo_mapping.index(parts[-4])
        lv_embed = lv_mapping.index(parts[-3])
        data = np.load(data_path).astype(np.float32)
        return  torch.from_numpy(data), torch.tensor(emo_embed), torch.tensor(lv_embed)

    def __len__(self):
        return len(self.data_path)
        # return 32
        
class A2LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 50
        self.fc0 = nn.Linear(self.input_size, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc31 = nn.Linear(32, 8)
        self.fc4 = nn.Linear(64, 32)
        self.fc41 = nn.Linear(32, 3)
        
        self.train_dataset = FaceDataset(args.train_dataset_path)
        self.val_dataset = FaceDataset(args.val_dataset_path)
        
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        
        self.emoloss = nn.CrossEntropyLoss()
        self.lvloss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr = args.learning_rate)
        
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
    
    def forward(self, audio):
        #audio: batch_size, dim
        output = torch.relu(self.fc0(audio))
        output = torch.relu(self.fc1(output))
        output = torch.relu(self.fc2(output))
        output1 = torch.relu(self.fc3(output))
        output1 = self.fc31(output1)
        output2 = torch.relu(self.fc4(output))
        output2 = self.fc41(output2)
        return output1, output2
        
        
    def train_all(self):
        if torch.cuda.is_available():
            self.cuda()
        #Load pretrain
        current_epoch = 0
        if args.use_pretrain == True:
            current_epoch = self.load_model() + 1
        
        train_loss = []
        val_loss = []
        best_running_loss = -1
        for epoch in range(current_epoch, args.n_epoches):
            print(f'\nTrain epoch {epoch}:\n')
            train_running_loss = self.train_epoch(epoch)
            train_loss.append(train_running_loss)
            
            print(f'\nValidate epoch {epoch}:\n')
            val_running_loss = self.validate_epoch(epoch)
            val_loss.append(val_running_loss)
            
            msg = f"\n| Epoch: {epoch}/{args.n_epoches} | Train Loss: {train_running_loss:#.4} | Val Loss: {val_running_loss:#.4} |"
            print(msg)
            
            if epoch % 50 == 0:
                ct = datetime.datetime.now()
                save_model(self, epoch, self.optimizer, f'{args.save_path}/e{epoch}-{ct}.pt')
            
            #Save best model
            if best_running_loss == -1 or val_running_loss < best_running_loss:
                print(f"\nSave the best model (epoch: {epoch})\n")
                save_model(self, epoch, self.optimizer, f'{args.save_path}/best_model.pt')
                best_running_loss = val_running_loss
        #Save last model
        print(f"\nSave the last model (epoch: {epoch})\n")
        save_model(self, epoch, self.optimizer, f'{args.save_path}/last_model.pt')

        #Save plot
        save_plots([], [], train_loss, val_loss, args.save_path)
        
    def train_epoch(self, epoch):
        self.train()
        running_loss = 0
        for step, (audio, emo_gt, lv_gt) in enumerate(self.train_dataloader):            
            if torch.cuda.is_available():
                audio, emo_gt, lv_gt = audio.cuda(), emo_gt.cuda(), lv_gt.cuda()      
             
            emo_pred, lv_pred = self(audio)   #1,25,68*2
            
            self.optimizer.zero_grad()
            emo_loss = self.emoloss(emo_pred, emo_gt)
            lv_loss = self.lvloss(lv_pred, lv_gt)
            loss = emo_loss + lv_loss * 0.1
            loss.backward()
            self.optimizer.step()
        
            running_loss += loss.item()
            msg = f"\r| Step: {step}/{len(self.train_dataloader)} of epoch {epoch} | Train Loss: {loss:#.4} |"
            sys.stdout.write(msg)
            sys.stdout.flush()
            
        return running_loss / len(self.train_dataloader)

    def validate_epoch(self, epoch):
        self.eval()
        running_loss = 0
        with torch.no_grad():
            for step, (audio, emo_gt, lv_gt) in enumerate(self.val_dataloader):
                if torch.cuda.is_available():
                    audio, emo_gt, lv_gt = audio.cuda(), emo_gt.cuda(), lv_gt.cuda()      
 
                emo_pred, lv_pred = self(audio)   #1,25,68*2
                
                self.optimizer.zero_grad()
                emo_loss = self.emoloss(emo_pred, emo_gt)
                lv_loss = self.lvloss(lv_pred, lv_gt)
                loss = emo_loss + lv_loss * 0.1     
                
                running_loss += loss.item()
                msg = f"\r| Step: {step}/{len(self.val_dataloader)} of epoch {epoch} | Val Loss: {loss:#.4} |"
                sys.stdout.write(msg)
                sys.stdout.flush()
        return running_loss / len(self.val_dataloader)
    
    def load_model(self, filename='best_model.pt'):
        return load_model(self, self.optimizer, save_file=f'{args.save_path}/{filename}')
            
            
if __name__ == '__main__': 
    net = A2LM()
    if args.train:
        net.train_all()
    else:
        net.load_model()
        net.validate_epoch()

