import json
import argparse
import numpy as np
import sys
import random
import datetime
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from modules.net_module import conv2d, DownBlock2d
from modules.util_module import *
from GrayImage_VAE import GrayImage_VAE

parser = argparse.ArgumentParser()
parser.add_argument('--train_dataset_path', type=str, default='data/train_MEAD.json')
parser.add_argument('--val_dataset_path', type=str, default='data/val_MEAD.json')

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=1.0e-4)
parser.add_argument('--n_epoches', type=int, default=50)

parser.add_argument('--save_path', type=str, default='result_A2GIWSI')
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
        parts = self.data_path[index].split('|') 
        # if args.train:       
        data = read_data_from_path(mfcc_path=parts[0], face_path = parts[2], start=parts[3], end=parts[4])
        # else:
        #     data = read_data_from_path(mfcc_path=parts[0], face_path = parts[2])
        return torch.from_numpy(data['mfcc_data_list']), torch.from_numpy(data['face_data_list'])

    def __len__(self):
        return len(self.data_path)
    
    def get_item_path(self, index):
        parts = self.data_path[index].split('|')
        return parts[2]
    
class Audio2GrayImageWithSampleImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = DownBlock2d(512,256)
        
        with torch.no_grad():
            self.first_image = read_data_from_path(face_path='/root/Datasets/Features/M003/faces/neutral/level_1/00001')
            self.first_image = torch.from_numpy(self.first_image['face_data_list'])
            self.grayimage_model = GrayImage_VAE()
            self.grayimage_model.load_model()
            if torch.cuda.is_available():
                self.first_image = self.first_image.cuda()
                self.grayimage_model.cuda()
                                
        self.audio_eocder = nn.Sequential(
            conv2d(1,64,3,1,1),
            conv2d(64,128,3,1,1),
            nn.MaxPool2d(3, stride=(1,2)),
            conv2d(128,256,3,1,1),
            conv2d(256,256,3,1,1),
            conv2d(256,512,3,1,1),
            nn.MaxPool2d(3, stride=(2,2))
            )
        self.audio_eocder_fc = nn.Sequential(
            nn.Linear(1024 *12,2048),
            nn.ReLU(True),
            nn.Linear(2048,256),
            nn.ReLU(True),

            )
        self.lstm = nn.LSTM(256+256,256,3,batch_first = True)
    
        self.decon = nn.Sequential(
                nn.ConvTranspose2d(256, 256, kernel_size=6, stride=2, padding=1, bias=True),#4,4
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),#8,8
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True), #16,16
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=True),#32,32
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=True),#64,64
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1, bias=True),#128,128
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1, bias=True),#256,256
            )
        
        self.train_dataset = FaceDataset(args.train_dataset_path)
        self.val_dataset = FaceDataset(args.val_dataset_path)
        
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr = args.learning_rate)
        
        self.init_model()
        self.num_params()
    
    def forward(self, audio):
        hidden = ( torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).cuda()),
                      torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).cuda()))
        
        lstm_input = []
        deco_out = []
        for step_t in range(audio.size(1)):
            current_audio = audio[ : ,step_t , :, :].unsqueeze(1)   #1,1,28,12
            current_feature = self.audio_eocder(current_audio)      #1,512,12,2
            current_feature = current_feature.view(current_feature.size(0), -1) #1,512*12*2
            current_feature = self.audio_eocder_fc(current_feature) #1,256
            
            with torch.no_grad():
                if step_t == 0:
                    next_frame = self.first_image   #1,256,256
                else:
                    next_frame = deco_out[-1].squeeze(1) #1,256,256
                    
                image_feature = self.grayimage_model.extract_feature(next_frame)      #1,512,2,2  
            image_feature = self.down(image_feature)                         #1,256,1,1  
            image_feature = image_feature.view(image_feature.shape[0], -1) #1,256
                        
            features = torch.cat([image_feature, current_feature], 1)      #1,512
            lstm_input.append(features)
            lstm_input_torch = torch.stack(lstm_input, dim = 1)
            lstm_out, _ = self.lstm(lstm_input_torch, hidden)                    #1,Step,256

            fc_in = lstm_out[:,step_t,:]                                    #1,256
            fc_feature = torch.unsqueeze(fc_in,2)
            fc_feature = torch.unsqueeze(fc_feature,3)                      #1,256,1,1
            decon_feature = self.decon(fc_feature)                          #1,1,256,256
            deco_out.append(decon_feature)

        deco_out = torch.stack(deco_out,dim=1)
        return deco_out
        
    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
            
    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print("Trainable Parameters: %.3fM" % parameters)
        return parameters 
        
    def train_all(self):
        if torch.cuda.is_available():
            self.cuda()
        #Load pretrain
        current_epoch = 0
        if args.use_pretrain == True:
            current_epoch = load_model(self, self.optimizer) + 1
        
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
        for step, (x,y) in enumerate(self.train_dataloader):            
            if torch.cuda.is_available():
                x,y = x.cuda(), y.cuda()    #x = 1,25,28,12; y = 1,25,256,256
            
            pred = self(x)                  #pred: 1,25,1,256,256
            pred = pred.squeeze(2)          #1,25,256,256
            
            self.optimizer.zero_grad()
            loss = self.criterion(pred, y)
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
            for step, (x,y) in enumerate(self.val_dataloader):
                if torch.cuda.is_available():
                    x,y = x.cuda(), y.cuda()    #y = 1,1,256,256
                
                pred = self(x)
                pred = pred.squeeze(2)
                loss = self.criterion(pred, y)      
                
                running_loss += loss.item()
                msg = f"\r| Step: {step}/{len(self.val_dataloader)} of epoch {epoch} | Val Loss: {loss:#.4} |"
                sys.stdout.write(msg)
                sys.stdout.flush()
        return running_loss / len(self.val_dataloader)
    
    def load_model(self, filename='best_model.pt'):
        load_model(self, self.optimizer, save_file=f'{args.save_path}/{filename}')
        
    def inference(self):
        if torch.cuda.is_available():
            self.cuda()
        with torch.no_grad():
            rand_index = random.choice(range(len(self.val_dataloader)))
            x,y = self.val_dataset[rand_index]      #x = 25,28,12; y = 25,256,256
            x = x.unsqueeze(0)
            if torch.cuda.is_available():
                x,y = x.cuda(), y.cuda()    
            pred = self(x)          #pred: 1,25,1,256,256
            pred = pred.squeeze(2)
            y = y.unsqueeze(0)        
            loss = self.criterion(pred, y)      
            print(f'Loss: {loss}')
            print(pred.shape)
            videoframes = pred[0].cpu().detach().numpy()
            videoframes =(videoframes * 255).astype(np.uint8)
            create_video(videoframes,f'{args.save_path}/prediction.mp4')
            
            
if __name__ == '__main__': 
    net = Audio2GrayImageWithSampleImage()
    if args.train:
        net.train_all()
    else:
        net.load_model()
        net.inference()
        
