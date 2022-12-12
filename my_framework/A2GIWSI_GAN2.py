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

parser = argparse.ArgumentParser()
parser.add_argument('--train_dataset_path', type=str, default='data/train_MEAD.json')
parser.add_argument('--val_dataset_path', type=str, default='data/val_MEAD.json')

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--generator_lr', type=float, default=1.0e-4)
parser.add_argument('--discriminator_lr', type=float, default=1.0e-4)
parser.add_argument('--n_epoches', type=int, default=50)

parser.add_argument('--save_path', type=str, default='result_A2GIWSI_GAN2')
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
        return len(self.data_path)
        # return 10
    
    def get_item_path(self, index):
        parts = self.data_path[index].split('|')
        return parts[2]

class A2GIWSI_Generator(nn.Module):
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
        deco_out = []
        for step_t in range(audio.size(1)):
            current_audio = audio[ : ,step_t , :, :].unsqueeze(1)   #1,1,28,12
            current_feature = self.audio_eocder(current_audio)      #1,512,12,2
            current_feature = current_feature.view(current_feature.size(0), -1) #1,512*12*2
            current_feature = self.audio_eocder_fc(current_feature) #1,256
            lstm_input.append(current_feature)
                
        lstm_input_torch = torch.stack(lstm_input, dim = 1)
        lstm_out, _ = self.lstm(lstm_input_torch, hidden)                    #1,Step,256
        for step_t in range(audio.size(1)):
            fc_in = lstm_out[:,step_t,:]                                    #1,256
            fc_feature = torch.unsqueeze(fc_in,2)
            fc_feature = torch.unsqueeze(fc_feature,3)                      #1,256,1,1
            decon_feature = self.decon(fc_feature)                          #1,1,256,256
            deco_out.append(decon_feature)

        deco_out = torch.stack(deco_out,dim=1)                              #1,25,256,256
        return deco_out
                    

class A2GIWSI_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        encoder_params = [
            ('same', 1, 64),    #(256,256,64)
            ('down', 64, 64),   #(128,128,64)
            ('down', 64, 128),  #(64,64,128)
            ('down', 128, 256), #(32,32,256)
            ('down', 256, 512), #(16,16,512)
            ('down', 512, 512), #(8,8,512)
            ('down', 512, 512), #(4,4,512)
            ('down', 512, 512), #(2,2,512)
            ('down', 512, 512), #(1,1,512)
        ]
        down_blocks = []
        for kernel in encoder_params:
            if kernel[0] == 'same':
                down_blocks.append(SameBlock2d(*kernel[1:]))  
            else:  
                down_blocks.append(DownBlock2d(*kernel[1:]))
        self.down_blocks = nn.ModuleList(down_blocks)
        
        self.conv_block = nn.Conv2d(512, 1, kernel_size=1)
        
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
              
    def forward(self, image):
        for i in range(len(self.down_blocks)):
            layer = self.down_blocks[i]
            if i == 0:
                out_encoder = layer(image)
            else:
                out_encoder = layer(out_encoder)    #25,512,1,1
        
        out = self.conv_block(out_encoder)  #25,1,1,1
        out = out.reshape(out.shape[0],-1)  #25,1
        
        return out_encoder, out        
    
            
class A2GIWSI_GAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = A2GIWSI_Generator()
        self.discriminator = A2GIWSI_Discriminator()
        if torch.cuda.is_available():
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
        
        self.train_dataset = FaceDataset(args.train_dataset_path)
        self.val_dataset = FaceDataset(args.val_dataset_path)
        
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.ssiml1_loss = MS_SSIM_L1_LOSS()
        self.l1_loss = nn.L1Loss()
        # self.vgg_loss = VGGLoss()
        self.generator_optimizer = optim.Adam(self.parameters(), lr = args.generator_lr)
        self.discriminator_optimizer = optim.Adam(self.parameters(), lr = args.discriminator_lr)
        
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
            
    def train_all(self):
        if torch.cuda.is_available():
            self.cuda()
        #Load pretrain
        current_epoch = 0
        if args.use_pretrain == True:
            current_epoch = load_model(self, self.optimizer) + 1
        
        train_G_loss = []
        train_D_loss = []
        val_G_loss = []
        val_D_loss = []
        best_running_loss = -1
        for epoch in range(current_epoch, args.n_epoches):
            print(f'\nTrain epoch {epoch}:\n')
            train_G_Loss_running, train_D_Loss_running = self.train_epoch(epoch)
            train_G_loss.append(train_G_Loss_running)
            train_D_loss.append(train_D_Loss_running)
            
            print(f'\nValidate epoch {epoch}:\n')
            val_G_Loss_running, val_D_Loss_running = self.validate_epoch(epoch)
            val_G_loss.append(val_G_Loss_running)
            val_D_loss.append(val_D_Loss_running)
            
            msg = f"\n| Epoch: {epoch}/{args.n_epoches} | Train G-Loss: {train_G_Loss_running:#.4}, D-Loss: {train_D_Loss_running:#.4} | Val G-Loss: {val_G_Loss_running:#.4}, D-Loss: {val_D_Loss_running:#.4} |"
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
        save_plots(train_G_loss, val_G_loss, train_D_loss, val_D_loss, args.save_path)
        
    def train_epoch(self, epoch):
        self.train()        
        G_Loss = 0
        D_Loss = 0
        for step, (audio,real_img) in enumerate(self.train_dataloader):            
            if torch.cuda.is_available():
                audio,real_img = audio.cuda(), real_img.cuda()    #x = 1,25,28,12; y = 1,25,256,256
            
            #train discriminator
            self.generator.eval()
            self.discriminator.train()
            
            with torch.no_grad():
                fake_img = self.generator(audio)    #1,25,1,256,256
            fake_img = fake_img.squeeze(2)      #1,25,256,256
            fake_img = fake_img.reshape(-1,1,fake_img.shape[2],fake_img.shape[3])   #25,1,256,256                
            real_img = real_img.reshape(-1,1,real_img.shape[2],real_img.shape[3])   #25,1,256,256
            
            fake_feature, fake_label = self.discriminator(fake_img)   #feature 25, 512,1,1; label 25,1
            real_feature, real_label = self.discriminator(real_img)   #feature 25, 512,1,1; label 25,1
                        
            ones = torch.ones(real_label.shape[0],1)
            zeros = torch.zeros(fake_label.shape[0],1)
            
            if torch.cuda.is_available():
                ones = ones.cuda()
                zeros = zeros.cuda()

            self.discriminator_optimizer.zero_grad()
            discriminator_real_loss = self.criterion(real_label, ones)            
            discriminator_fake_loss = self.criterion(fake_label, zeros)
            discriminator_loss = (discriminator_real_loss * 2 + discriminator_fake_loss) / 2
            discriminator_loss.backward()
            self.discriminator_optimizer.step()
            
            #train generator
            self.generator.train()
            self.discriminator.eval()
            
            fake_img = self.generator(audio)    #1,25,1,256,256
            fake_img = fake_img.squeeze(2)      #1,25,256,256
            fake_img = fake_img.reshape(-1,1,fake_img.shape[2],fake_img.shape[3])   #25,1,256,256
            
            fake_feature, fake_label = self.discriminator(fake_img)   #feature 25, 512,1,1; label 25,1
            real_feature, real_label = self.discriminator(real_img)   #feature 25, 512,1,1; label 25,1
            
            self.generator_optimizer.zero_grad()
            generator_bce_loss = self.criterion(fake_label, ones)
            generator_ssiml1_loss = self.ssiml1_loss(fake_img, real_img) * 100
            # generator_vgg_loss = self.vgg_loss(fake_img, real_img)
            generator_fm_loss = self.l1_loss(fake_feature, real_feature)
            generator_loss = generator_bce_loss + generator_ssiml1_loss + generator_fm_loss
            generator_loss.backward()
            self.generator_optimizer.step()
            
            #Summary        
            G_Loss += generator_loss.item()
            D_Loss += discriminator_loss.item()
            msg = f"\r| Step: {step}/{len(self.train_dataloader)} of epoch {epoch} | Loss_D: {discriminator_loss.item():#.4} | Loss_G: {generator_loss.item():#.4} |"
            sys.stdout.write(msg)
            sys.stdout.flush()
            
        return G_Loss / len(self.train_dataloader), D_Loss / len(self.train_dataloader)

    def validate_epoch(self, epoch):
        self.eval()
        G_Loss = 0
        D_Loss = 0
        for step, (audio,real_img) in enumerate(self.val_dataloader):            
            if torch.cuda.is_available():
                audio,real_img = audio.cuda(), real_img.cuda()    #x = 1,25,28,12; y = 1,25,256,256
                
            self.generator.eval()
            self.discriminator.eval()
            
            fake_img = self.generator(audio)    #1,25,1,256,256
            fake_img = fake_img.squeeze(2)      #1,25,256,256
            
            fake_img = fake_img.reshape(-1,1,fake_img.shape[2],fake_img.shape[3])   #25,1,256,256
            real_img = real_img.reshape(-1,1,real_img.shape[2],real_img.shape[3])
            
            fake_feature, fake_label = self.discriminator(fake_img)   #feature 25, 512,1,1; label 25,1
            real_feature, real_label = self.discriminator(real_img)   #feature 25, 512,1,1; label 25,1
            
            ones = torch.ones(real_label.shape)
            zeros = torch.zeros(fake_label.shape)
            if torch.cuda.is_available():
                ones = ones.cuda()
                zeros = zeros.cuda()
            discriminator_real_loss = self.criterion(real_label, ones)
            discriminator_fake_loss = self.criterion(fake_label, zeros)
            discriminator_loss = (discriminator_real_loss * 2 + discriminator_fake_loss)/2

            generator_bce_loss = self.criterion(fake_label, ones)
            generator_ssiml1_loss = self.ssiml1_loss(fake_img, real_img)
            generator_fm_loss = self.l1_loss(fake_feature, real_feature)
            generator_loss = generator_bce_loss + generator_ssiml1_loss + generator_fm_loss
            
            #Summary        
            G_Loss += generator_loss.item()
            D_Loss += discriminator_loss.item()
            msg = f"\r| Step: {step}/{len(self.train_dataloader)} of epoch {epoch} | Loss_D: {discriminator_loss.item():#.4} | Loss_G: {generator_loss.item():#.4} |"
            sys.stdout.write(msg)
            sys.stdout.flush()
        return G_Loss / len(self.train_dataloader), D_Loss / len(self.train_dataloader)
    
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
        
