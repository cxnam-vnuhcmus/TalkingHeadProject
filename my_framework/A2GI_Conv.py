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

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=1.0e-4)
parser.add_argument('--n_epoches', type=int, default=200)

parser.add_argument('--save_path', type=str, default='result_A2GI_Conv')
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
        data = read_data_from_path(mfcc_path=parts[0], lm_path=parts[1], face_path = parts[2], start=parts[3], end=parts[4])
        # else:
        #     data = read_data_from_path(mfcc_path=parts[0], face_path = parts[2])
        lms = data['lm_data_list']
        imgs = data['face_data_list']
        
        results = []
        for i in range(len(imgs)):
            lm = lms[i].reshape(68,2).astype(int)
            result = calculate_segmap(imgs[i], lm)
            result = torch.from_numpy(result.reshape(-1))
            results.append(result)
        results = torch.stack(results)
        return torch.from_numpy(data['mfcc_data_list']), results

    def __len__(self):
        return len(self.data_path)
        # return 10
    
class A2GI_Conv(nn.Module):
    def __init__(self):
        super().__init__()
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
            nn.Linear(2048,512),
            nn.ReLU(True),
            )
        self.lstm = nn.LSTM(512,256,3,batch_first = True)
        
        self.fc = nn.Sequential(
                nn.Linear(in_features=256, out_features=512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.2),
                nn.Linear(1024, 64*64))
        
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
            print(self.__class__.__name__ + " Parameters: %.3fM" % parameters)
        return parameters 
    
    def forward(self, audio):
        hidden = ( torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).cuda()),
                      torch.autograd.Variable(torch.zeros(3, audio.size(0), 256).cuda()))
        
        lstm_input = []
        for step_t in range(audio.size(1)):
            current_audio = audio[ : ,step_t , :, :].unsqueeze(1)   #1,1,28,12
            current_feature = self.audio_eocder(current_audio)      #1,512,12,2
            current_feature = current_feature.view(current_feature.size(0), -1) #1,512*12*2
            current_feature = self.audio_eocder_fc(current_feature) #1,256
            lstm_input.append(current_feature)
            
        lstm_input_torch = torch.stack(lstm_input, dim = 1)         #1,F,256
        lstm_out, _ = self.lstm(lstm_input_torch, hidden)           #1,F,256
        
        lstm_out = lstm_out.reshape(-1,lstm_out.shape[-1])             #1*F,256
        lm_pred = self.fc(lstm_out)                                 #1*F,64*64
        lm_pred = lm_pred.view(audio.shape[0], audio.shape[1],-1)   #1,F,64*64

        # lm_pred = torch.sigmoid(lm_pred)
        # lm_pred[lm_pred >= 0.5] = 1
        # lm_pred[lm_pred < 0.5] = 0
        return lm_pred
        
        
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
                audio, lm_gt = audio.cuda(), lm_gt.type(torch.cuda.FloatTensor).cuda()      #audio = 1,25,28,12; lm = 1,25,64*64
                lm_gt[lm_gt < 10] = 0.
                lm_gt[lm_gt >= 10] = 1.
                # lm_gt = torch.autograd.Variable(lm_gt, requires_grad=True).cuda()
                
            lm_pred = self(audio)   #1,25,64*64

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
                    audio, lm_gt = audio.cuda(), lm_gt.cuda()      #audio = 1,25,28,12; lm = 1,25,64*64
                    
                lm_pred = self(audio)       #1,25,64*64
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
            audio = audio.unsqueeze(0)    #x = 1,25,28,12; y = 25,68*2
            if torch.cuda.is_available():
                audio,lm_gt = audio.cuda(), lm_gt.cuda()   
            
            lm_pred = self(audio)           #1,25,68*2
            lm_pred = lm_pred.squeeze(0)    #25,68*2
            loss = self.criterion(lm_pred, lm_gt)      
            print(f'Loss: {loss}')
            
            lm_pred = lm_pred.reshape(-1,64,64)
            lm_pred = lm_pred.cpu().detach().numpy()
            
            lm_gt = lm_gt.reshape(lm_gt.shape[0],64,64)
            lm_gt = lm_gt.cpu().detach().numpy()
            
            outputs = []
            for i in range(len(lm_pred)):
                result_img = np.zeros((64, 64*2))
                result_img[:,:64] = lm_gt[i]
                result_img[:,64:] = lm_pred[i]
                outputs.append(result_img)
            
            create_video(outputs,f'{args.save_path}/prediction.mp4', fps=10)
    
    def load_model(self, filename='best_model.pt'):
        return load_model(self, self.optimizer, save_file=f'{args.save_path}/{filename}')
            
            
if __name__ == '__main__': 
    net = A2GI_Conv()
    if args.train:
        net.train_all()
    else:
        net.load_model()
        net.inference()
