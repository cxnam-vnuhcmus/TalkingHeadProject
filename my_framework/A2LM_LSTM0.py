import json
import argparse
import numpy as np
import sys
import random
import datetime
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from modules.util_module import *
from modules.net_module import conv2d
from modules.face_visual_module import connect_face_keypoints

parser = argparse.ArgumentParser()
parser.add_argument('--train_dataset_path', type=str, default='data/train_MEAD.json')
parser.add_argument('--val_dataset_path', type=str, default='data/val_MEAD.json')

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=1.0e-4)
parser.add_argument('--n_epoches', type=int, default=500)

parser.add_argument('--save_path', type=str, default='result_A2LM_LSTM0')
parser.add_argument('--use_pretrain', type=bool, default=False)
parser.add_argument('--train', action='store_true')
args = parser.parse_args()

class FaceDataset(Dataset):
    def __init__(self, path=None):
        self.data_path = []
        if path is not None:
            with open(path, 'r') as f:
                self.data_path = json.load(f)

    def __getitem__(self, index):
        rand_index = random.choice(range(len(self.data_path)))
        parts = self.data_path[rand_index].split('|')        

        # if args.train:
        data = read_data_from_path(mfcc_path=parts[0], lm_path=parts[1], start=parts[3], end=parts[4])
        # else:
        #     data = read_data_from_path(mfcc_path=parts[0], lm_path=parts[1])
        return  torch.from_numpy(data['mfcc_data_list']), torch.from_numpy(data['lm_data_list'])

    def __len__(self):
        return len(self.data_path)
        # return 1000
        
class A2LM_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(
                nn.Linear(in_features=28*12, out_features=512),
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
                nn.Linear(512, 68*2))
        
        self.train_dataset = FaceDataset(args.train_dataset_path)
        self.val_dataset = FaceDataset(args.val_dataset_path)
        
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        
        self.criterion = nn.MSELoss()
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
        for step, (audio, lm_gt) in enumerate(self.train_dataloader):            
            if torch.cuda.is_available():
                audio, lm_gt = audio.cuda(), lm_gt.cuda()      #audio = 1,25,28,12; lm = 1,25,68*2
                audio = audio.reshape(audio.shape[0], audio.shape[1], -1)   #1,25,28*12
                
            lm_pred = self(audio)   #1,25,68*2
            
            self.optimizer.zero_grad()
            loss = self.criterion(lm_pred, lm_gt)
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
            for step, (audio, lm_gt) in enumerate(self.val_dataloader):
                if torch.cuda.is_available():
                    audio, lm_gt = audio.cuda(), lm_gt.cuda()      #audio = 1,25,28,12; lm = 1,25,68*2
                    audio = audio.reshape(audio.shape[0], audio.shape[1], -1)   #1,25,28*12
                    
                lm_pred = self(audio)       #1,25,68*2
                loss = self.criterion(lm_pred, lm_gt)      
                
                running_loss += loss.item()
                msg = f"\r| Step: {step}/{len(self.val_dataloader)} of epoch {epoch} | Val Loss: {loss:#.4} |"
                sys.stdout.write(msg)
                sys.stdout.flush()
        return running_loss / len(self.val_dataloader)
    
    def inference(self):
        if torch.cuda.is_available():
            self.cuda()
        with torch.no_grad():
            rand_index = random.choice(range(len(self.val_dataloader)))
            audio,lm_gt = self.val_dataset[rand_index]      
            audio = audio.unsqueeze(0)    #x = 1,25,28,12; y = 1,25,68*2
            audio = audio.reshape(audio.shape[0], audio.shape[1], -1)   #1,25,28*12
            if torch.cuda.is_available():
                audio,lm_gt = audio.cuda(), lm_gt.cuda()   
            
            lm_pred = self(audio)           #1,25,68*2
            lm_pred = lm_pred.squeeze(0)    #25,68*2
            loss = self.criterion(lm_pred, lm_gt)      
            print(f'Loss: {loss}')
            
            lm_pred = lm_pred.reshape(-1,68,2)
            lm_pred = lm_pred.cpu().detach().numpy()
            outputs_pred = connect_face_keypoints(256,256,lm_pred)
            
            lm_gt = lm_gt.reshape(lm_gt.shape[0],68,2)
            lm_gt = lm_gt.cpu().detach().numpy()
            outputs_gt = connect_face_keypoints(256,256,lm_gt)
            
            outputs = []
            for i in range(len(outputs_gt)):
                result_img = np.zeros((256, 256*2, 1))
                result_img[:,:256,:] = outputs_gt[i] * 255
                result_img[:,256:,:] = outputs_pred[i] * 255
                outputs.append(result_img)
            
            create_video(outputs,f'{args.save_path}/prediction.mp4',fps=10)
    
    def load_model(self, filename='best_model.pt'):
        return load_model(self, self.optimizer, save_file=f'{args.save_path}/{filename}')
            
            
if __name__ == '__main__': 
    net = A2LM_LSTM()
    if args.train:
        net.train_all()
    else:
        net.load_model()
        net.inference()
