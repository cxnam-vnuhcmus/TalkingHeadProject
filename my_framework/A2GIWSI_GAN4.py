import json
import argparse
import numpy as np
import sys
import random
import datetime
import torch
import torchvision
import torchvision.transforms.functional as fn
from torch import nn, optim, sigmoid
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from modules.net_module import conv2d, SameBlock2d, DownBlock2d, UpBlock2d
from modules.util_module import *
from evaluation.evaluation_image import MS_SSIM_L1_LOSS

parser = argparse.ArgumentParser()
parser.add_argument('--train_dataset_path', type=str, default='data/train_MEAD.json')
parser.add_argument('--val_dataset_path', type=str, default='data/val_MEAD.json')

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--generator_lr', type=float, default=1.0e-4)
parser.add_argument('--discriminator_lr', type=float, default=1.0e-4)
parser.add_argument('--n_epoches', type=int, default=50)
parser.add_argument('--n_scale', type=int, default=3)
parser.add_argument('--base_size', type=int, default=32)

parser.add_argument('--save_path', type=str, default='result_A2GIWSI_GAN4')
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
        # data = read_data_from_path(mfcc_path=parts[0], lm_path=parts[1], face_path = parts[2])
        
        lms = data['lm_data_list']
        imgs = data['face_data_list']
        
        image_pyramid_list = []
        for i in range(len(imgs)):
            lm = lms[i].reshape(68,2).astype(int)
            segmap = calculate_segmap(imgs[i], lm)            
            
            nlayer = [11,9,5,1]
            image_pyramid = []
            
            for i in range(args.n_scale):
                img_copy = segmap.copy()
                img_copy[img_copy < nlayer[i]] = 0
                img_copy[img_copy >= nlayer[i]] = 1
                img_copy = np.resize(img_copy, (args.base_size * (2**i),args.base_size * (2**i)))
                img_copy = np.expand_dims(img_copy, axis=0)  
                img_copy = np.expand_dims(img_copy, axis=0)  
                image_pyramid.append(torch.from_numpy(img_copy))      #1,1,32,32; 1,1,64,64; 1,1,128,128
            
            if len(image_pyramid_list) == 0:
                image_pyramid_list = image_pyramid
            else:
                for i in range(args.n_scale):
                    image_pyramid_list[i] = torch.vstack([image_pyramid_list[i], image_pyramid[i]]) #25,1,32,32; 25,1,64,64; 25,1,128,128
                    
        return torch.from_numpy(data['mfcc_data_list']), image_pyramid_list

    def __len__(self):
        # return len(self.data_path)
        return 1000
    
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
    
        self.audio_decode_fc_layers = []
        for i in range(args.n_scale,0,-1):
            if len(self.audio_decode_fc_layers) == 0:
                layer = nn.Sequential(
                    nn.Linear(256, args.base_size * (2**(i-1))),
                    nn.ReLU(True),
                )
            else:
                layer = nn.Sequential(
                    nn.Linear(args.base_size * (2**i), args.base_size * (2**(i-1))),
                    nn.ReLU(True),
                )
            self.audio_decode_fc_layers.append(layer)
        self.audio_decode_fc_layers = nn.ModuleList(self.audio_decode_fc_layers)
        
        self.segmap_decode_layers = []
        for i in range(args.n_scale,0,-1):
            if len(self.segmap_decode_layers) == 0:
                layer = SameBlock2d(1, args.base_size * (2**(i-1)))
            else:
                layer = UpBlock2d(args.base_size * (2**i) + 1, args.base_size * (2**(i-1)))
            self.segmap_decode_layers.append(layer)
            
            layer = SameBlock2d(args.base_size * (2**(i-1)), 1)
            self.segmap_decode_layers.append(layer)
            
        self.segmap_decode_layers = nn.ModuleList(self.segmap_decode_layers)
        
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
        
        gen_outs = []
        for step_t in range(audio.size(1)):
            current_audio = audio[ : ,step_t , :, :].unsqueeze(1)   #1,1,28,12
            current_feature = self.audio_eocder(current_audio)      #1,512,12,2
            current_feature = current_feature.view(current_feature.size(0), -1) #1,512*12*2
            current_feature = self.audio_eocder_fc(current_feature) #1,512
            current_feature = current_feature.unsqueeze(0)          #1,1,512
            lstm_out, hidden = self.lstm(current_feature, hidden)   #1,1,256
            
            lstm_out = lstm_out.view(-1, lstm_out.shape[1] * lstm_out.shape[2]) #1,256
            audio_outs = [lstm_out]
            for layer in self.audio_decode_fc_layers:
                out = layer(audio_outs[-1])
                audio_outs.append(out)  
            audio_outs.reverse()    #1,32; 1,64; 1,128; 1,256
            
            noise = torch.rand(1,1,32,32)
            if torch.cuda.is_available():
                noise = noise.cuda()
            
            segmap_out_up = []
            segmap_out_down = []
            for i in range(0, len(self.segmap_decode_layers), 2): 
                up = self.segmap_decode_layers[i]
                down = self.segmap_decode_layers[i+1]
                if len(segmap_out_down) == 0:
                    out_up = up(noise)    
                    out_down = down(out_up)
                else:
                    out_up = up(torch.cat([segmap_out_up[-1], segmap_out_down[-1]], dim=1))    
                    out_down = down(out_up)
                segmap_out_up.append(out_up)        #1,128,32,32 -> 1,64,64,64 -> 1,32,128,128
                segmap_out_down.append(out_down)    #1,1,32,32 -> 1,1,64,64 -> 1,1,128,128
            
            if step_t == 0:
                gen_outs = segmap_out_down
            else:
                for i in range(len(segmap_out_down)):
                    gen_outs[i] = torch.vstack([gen_outs[i], segmap_out_down[i]])   #25,1,32,32 -> 25,1,64,64 -> 25,1,128,128 ->25,1,256,256       
        return gen_outs
                    

class A2GIWSI_Discriminator(nn.Module):
    def __init__(self, input_shape):    #32,128,128; 64,64,64; 128,32,32
        super().__init__()
        
        self.down_blocks = []
        channel,size = input_shape[0], input_shape[1]
        while(size > args.base_size):
            layer = DownBlock2d(1, 1)
            self.down_blocks.append(layer)
            size = size // 2            
        self.down_blocks = nn.ModuleList(self.down_blocks)

        self.fc_blocks = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(True),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, 256),
                nn.ReLU(True),
                nn.Linear(256, 256),
                nn.ReLU(True),
                nn.Linear(256, 1),
                nn.Sigmoid()
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
              
    def forward(self, image):
        out = image
        for i in range(len(self.down_blocks)):  #25,1,X,X
            layer = self.down_blocks[i]
            out = layer(out)
        out = out.view(out.shape[0],-1)         #25,1*32*32
        for i in range(len(self.fc_blocks)):    
            layer = self.fc_blocks[i]
            out = layer(out)
        return out                              #25,1
    
class A2GIWSI_MultiPatch_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        discriminators = []
        for i in range(args.n_scale):
            discriminator = A2GIWSI_Discriminator((32*(2**i),32*(2**(args.n_scale-1))//(2**i),32*(2**(args.n_scale-1))//(2**i)))
            discriminators.append(discriminator)
        discriminators.reverse()
        self.discriminators = nn.ModuleList(discriminators)
        
    def forward(self, image):
        discriminator_out = []
        for i in range(args.n_scale):
            layer = self.discriminators[i]
            out = layer(image[i])
            discriminator_out.append(out)
        return discriminator_out
        
class A2GIWSI_GAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = A2GIWSI_Generator()
        self.discriminator = A2GIWSI_MultiPatch_Discriminator()
        if torch.cuda.is_available():
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
        
        print("init dataset")
        self.train_dataset = FaceDataset(args.train_dataset_path)
        self.val_dataset = FaceDataset(args.val_dataset_path)
        
        print("init dataloader")
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        
        print("init loss and optim")
        self.criterion = nn.BCELoss()
        self.ssiml1_loss = MS_SSIM_L1_LOSS()
        self.generator_optimizer = optim.Adam(self.parameters(), lr = args.generator_lr)
        self.discriminator_optimizer = optim.Adam(self.parameters(), lr = args.discriminator_lr)
            
        print("init model")
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
                audio = audio.cuda()     #1,25,28,12
                for i in range(len(real_img)):
                    real_img[i] = real_img[i].reshape(-1,*real_img[i].shape[2:])
                    real_img[i] = real_img[i].type(torch.FloatTensor)
                    real_img[i] = torch.autograd.Variable(real_img[i], requires_grad=True).cuda()  #25,1,32,32; 25,1,64,64; 25,1,128,128
            
            #train discriminator
            self.generator.eval()
            self.discriminator.train()
            
            with torch.no_grad():
                gen_outs = self.generator(audio)    #25,1,32,32 -> 25,1,64,64 -> 25,1,128,128 ->25,1,256,256       
            
            fake_label = self.discriminator(gen_outs)
            real_label = self.discriminator(real_img)  
                                
            self.discriminator_optimizer.zero_grad()       
            discriminator_loss = 0
            for i in range(args.n_scale):
                value = (1 - real_label[i]) ** 2 +  fake_label[i] ** 2
                discriminator_loss += value.mean()*1000
            discriminator_loss.backward()
            self.discriminator_optimizer.step()
            
            #train generator
            self.generator.train()
            self.discriminator.eval()
            
            gen_outs = self.generator(audio)   
            fake_label = self.discriminator(gen_outs)
            
            self.generator_optimizer.zero_grad()
            generator_loss = 0
            for i in range(args.n_scale):
                gan_loss = (1 - fake_label[i]) ** 2
                ssim_loss = self.ssiml1_loss(gen_outs[i], real_img[i])
                generator_loss += gan_loss.mean()*100 + ssim_loss*100
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
                audio = audio.cuda()     #1,25,28,12
                for i in range(len(real_img)):
                    real_img[i] = real_img[i].reshape(-1,*real_img[i].shape[2:])
                    real_img[i] = real_img[i].type(torch.FloatTensor)
                    real_img[i] = torch.autograd.Variable(real_img[i], requires_grad=True).cuda()  #25,1,32,32; 25,1,64,64; 25,1,128,128
                        
            #generator
            self.generator.eval()
            self.discriminator.eval()
            
            gen_outs = self.generator(audio)    #25,1,32,32 -> 25,1,64,64 -> 25,1,128,128 ->25,1,256,256       
            
            fake_label = self.discriminator(gen_outs)
            real_label = self.discriminator(real_img)  
                
            discriminator_loss = 0
            for i in range(args.n_scale):
                value = (1 - real_label[i]) ** 2 +  fake_label[i] ** 2
                discriminator_loss += value.mean()*1000
            
            generator_loss = 0
            for i in range(args.n_scale):
                gan_loss = fake_label[i] ** 2
                ssim_loss = self.ssiml1_loss(gen_outs[i], real_img[i])
                generator_loss += gan_loss.mean()*100 + ssim_loss*100
            
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
            audio = audio.unsqueeze(0)    #x = 1,25,28,12; y = 1,25,1,256,256
            if torch.cuda.is_available():
                audio = audio.cuda()     #1,25,28,12
                for i in range(len(real_img)):
                    real_img[i] = real_img[i].reshape(-1,*real_img[i].shape[2:])
                    real_img[i] = real_img[i].type(torch.FloatTensor)
                    real_img[i] = torch.autograd.Variable(real_img[i], requires_grad=True).cuda()  #25,1,32,32; 25,1,64,64; 25,1,128,128
     

            gen_outs = self.generator(audio) #25,256,32,32 -> 25,128,64,64 -> 25,64,128,128 -> 25,32,256,256       
            
            videoframes = gen_outs[-1][0].cpu().detach().numpy()
            videoframes =(videoframes * 255).astype(np.uint8)
            create_video(videoframes,f'{args.save_path}/prediction.mp4')
            
            
if __name__ == '__main__': 
    net = A2GIWSI_GAN()
    if args.train:
        net.train_all()
    else:
        net.load_model()
        net.inference()
        
