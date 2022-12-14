import json
import argparse
import numpy as np
import sys
import random
import datetime
import torch
from torch import nn, optim, sigmoid
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from modules.net_module import conv2d, SameBlock2d, DownBlock2d
from modules.util_module import *
from evaluation.evaluation_image import MS_SSIM_L1_LOSS
from GrayImage_VAE import GrayImage_VAE

parser = argparse.ArgumentParser()
parser.add_argument('--train_dataset_path', type=str, default='data/train_MEAD.json')
parser.add_argument('--val_dataset_path', type=str, default='data/val_MEAD.json')

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--generator_lr', type=float, default=1.0e-4)
parser.add_argument('--discriminator_lr', type=float, default=1.0e-4)
parser.add_argument('--n_epoches', type=int, default=20)

parser.add_argument('--save_path', type=str, default='result_A2GIWSI_GAN3')
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
        data = read_data_from_path(mfcc_path=parts[0], face_path = parts[2], start=parts[3], end=parts[4])
        # else:
        #     data = read_data_from_path(mfcc_path=parts[0], face_path = parts[2])
        return torch.from_numpy(data['mfcc_data_list']), torch.from_numpy(data['face_data_list'])

    def __len__(self):
        # return len(self.data_path)
        return 1000
    
    def get_item_path(self, index):
        parts = self.data_path[index].split('|')
        return parts[2]
          
class A2GIWSI_GAN(nn.Module):
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
            nn.Linear(1024 * 12, 1024 * 6),
            # nn.BatchNorm1d(1024 * 6),
            nn.LeakyReLU(0.2),
            nn.Linear(1024 * 6, 1024 * 2),
            # nn.BatchNorm1d(1024 * 2),
            nn.LeakyReLU(0.2),
            )
        self.lstm = nn.LSTM(2048,2048,3,batch_first = True)
    
        with torch.no_grad():
            self.vae = GrayImage_VAE()
            self.vae.load_model()
        
        self.train_dataset = FaceDataset(args.train_dataset_path)
        self.val_dataset = FaceDataset(args.val_dataset_path)
        
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        self.ssiml1_loss = MS_SSIM_L1_LOSS()
        self.generator_optimizer = optim.Adam(self.parameters(), lr = args.generator_lr)
        
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
        hidden = ( torch.autograd.Variable(torch.zeros(3, audio.size(0), 2048).cuda()),
                      torch.autograd.Variable(torch.zeros(3, audio.size(0), 2048).cuda()))
        
        lstm_input = []
        for step_t in range(audio.size(1)):
            current_audio = audio[ : ,step_t , :, :].unsqueeze(1)   #1,1,28,12
            current_feature = self.audio_eocder(current_audio)      #1,512,12,2
            current_feature = current_feature.view(current_feature.size(0), -1) #1,512*12*2
            current_feature = self.audio_eocder_fc(current_feature) #1,512*2*2
            lstm_input.append(current_feature)
                
        lstm_input_torch = torch.stack(lstm_input, dim = 1)         #1,25,512*2*2
        lstm_out, _ = self.lstm(lstm_input_torch, hidden)           #1,25,512*2*2
        out = lstm_out.reshape(lstm_out.shape[0],lstm_out.shape[1],512,2,2)
        return out
            
    def train_all(self):
        if torch.cuda.is_available():
            self.cuda()
            self.vae = self.vae.cuda()
        #Load pretrain
        current_epoch = 0
        if args.use_pretrain == True:
            current_epoch = load_model(self, self.optimizer) + 1
        
        train_G_loss = []
        val_G_loss = []
        best_running_loss = -1
        for epoch in range(current_epoch, args.n_epoches):
            print(f'\nTrain epoch {epoch}:\n')
            train_G_Loss_running = self.train_epoch(epoch)
            train_G_loss.append(train_G_Loss_running)
            
            print(f'\nValidate epoch {epoch}:\n')
            val_G_Loss_running = self.validate_epoch(epoch)
            val_G_loss.append(val_G_Loss_running)
            
            msg = f"\n| Epoch: {epoch}/{args.n_epoches} | Train G-Loss: {train_G_Loss_running:#.4}| Val G-Loss: {val_G_Loss_running:#.4} |"
            print(msg)
            
            ct = datetime.datetime.now()
            save_model(self, epoch, None, f'{args.save_path}/e{epoch}-{ct}.pt')
            
            #Save best model
            if best_running_loss == -1 or val_G_Loss_running < best_running_loss:
                print(f"\nSave the best model (epoch: {epoch})\n")
                save_model(self, epoch, None, f'{args.save_path}/best_model.pt')
                best_running_loss = val_G_Loss_running
        #Save last model
        print(f"\nSave the last model (epoch: {epoch})\n")
        save_model(self, epoch, None, f'{args.save_path}/last_model.pt')

        #Save plot
        save_plots([], [], train_G_loss, val_G_loss, args.save_path)
        
    def train_epoch(self, epoch):
        self.train()        
        G_Loss = 0
        for step, (audio,real_img) in enumerate(self.train_dataloader):            
            if torch.cuda.is_available():
                audio,real_img = audio.cuda(), real_img.cuda()    #x = 1,25,28,12; y = 1,25,256,256
            
            fake_feature = self(audio)  #1,25,512,2,2
            fake_feature = fake_feature.reshape(-1, *fake_feature.shape[2:])
            with torch.no_grad():
                real_img = real_img.view(-1, 1, real_img.shape[2], real_img.shape[3]) #25,1,256,256
                real_feature = self.vae.extract_feature(real_img)   #1,25,512,2,2
                fake_img = self.vae.decoder(fake_feature)
                new_real_img = self.vae.decoder(real_feature)
            
            self.generator_optimizer.zero_grad()
            l1_loss = self.l1_loss(fake_feature, real_feature)
            ssim_loss = self.ssiml1_loss(fake_img, new_real_img) * 100
            generator_loss = l1_loss + ssim_loss
            generator_loss.backward()
            self.generator_optimizer.step()
            
            #Summary        
            G_Loss += generator_loss.item()
            msg = f"\r| Step: {step}/{len(self.train_dataloader)} of epoch {epoch} | Loss_G: {generator_loss.item():#.4} |"
            sys.stdout.write(msg)
            sys.stdout.flush()
            
        return G_Loss / len(self.train_dataloader)

    def validate_epoch(self, epoch):
        self.eval()
        G_Loss = 0
        for step, (audio,real_img) in enumerate(self.val_dataloader):            
            if torch.cuda.is_available():
                audio,real_img = audio.cuda(), real_img.cuda()    #x = 1,25,28,12; y = 1,25,256,256
                
            fake_feature = self(audio)  #1,25,512,2,2
            
            with torch.no_grad():
                real_feature = self.vae.extract_feature(real_img)   #1,25,512,2,2
            
            generator_loss = self.l1_loss(fake_feature, real_feature)
            
            #Summary        
            G_Loss += generator_loss.item()
            msg = f"\r| Step: {step}/{len(self.train_dataloader)} of epoch {epoch} | Loss_G: {generator_loss.item():#.4} |"
            sys.stdout.write(msg)
            sys.stdout.flush()
            
        return G_Loss / len(self.train_dataloader)
    
    def load_model(self, filename='best_model.pt'):
        load_model(self, None, save_file=f'{args.save_path}/{filename}')
        
    def inference(self):
        if torch.cuda.is_available():
            self.cuda()
        with torch.no_grad():
            rand_index = random.choice(range(len(self.val_dataloader)))
            audio,real_img = self.val_dataset[rand_index]      
            audio,real_img = audio.unsqueeze(0), real_img.unsqueeze(0)    #x = 1,25,28,12; y = 1,25,256,256
            if torch.cuda.is_available():
                audio,real_img = audio.cuda(), real_img.cuda()    

            fake_img = self.generator(audio)          #pred: 1,25,1,256,256
            fake_img = fake_img.squeeze(2)

            # loss = self.criterion(pred, y)      
            # print(f'Loss: {loss}')
            # print(pred.shape)
            
            videoframes = fake_img[0].cpu().detach().numpy()
            videoframes =(videoframes * 255).astype(np.uint8)
            create_video(videoframes,f'{args.save_path}/prediction.mp4')
            
            
if __name__ == '__main__': 
    net = A2GIWSI_GAN()
    if args.train:
        net.train_all()
    else:
        net.load_model()
        net.inference()
        
