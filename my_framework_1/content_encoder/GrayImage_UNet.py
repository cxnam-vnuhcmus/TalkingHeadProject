import json
import argparse
import numpy as np
import sys
import random
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from modules.net_module import UNet
from modules.util_module import *

parser = argparse.ArgumentParser()
parser.add_argument('--train_dataset_path', type=str, default='data/train_MEAD.json')
parser.add_argument('--val_dataset_path', type=str, default='data/val_MEAD.json')

parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=1.0e-4)
parser.add_argument('--n_epoches', type=int, default=100)

parser.add_argument('--save_best_model_path', type=str, default='result/best_model.pt')
parser.add_argument('--save_last_model_path', type=str, default='result/last_model.pt')
parser.add_argument('--save_plot_path', type=str, default='result')
parser.add_argument('--use_pretrain', type=bool, default=False)
args = parser.parse_args()

class FaceDataset(Dataset):
    def __init__(self, path=None):
        self.data_path = []
        if path is not None:
            with open(path, 'r') as f:
                self.data_path = json.load(f)

    def __getitem__(self, index):
        parts = self.data_path[index].split('|')        
        data = read_data_from_path(face_path = parts[2])
        return  torch.from_numpy(data['face_data_list'])

    def __len__(self):
        return len(self.data_path)
    
    def get_item_path(self, index):
        parts = self.data_path[index].split('|')
        return parts[2]
    
class GrayImage_UNet(UNet):
    def __init__(self):
        encoder_params = [
            ('same', 1, 64),    #(256,256,64)
            ('down', 64, 64),   #(128,128,64)
            ('down', 64, 128),  #(64,64,128)
            ('down', 128, 256), #(32,32,256)
            ('down', 256, 512), #(16,16,512)
            ('down', 512, 512), #(8,8,512)
            ('down', 512, 512), #(4,4,512)
            ('down', 512, 512), #(2,2,512)
        ]
        decoder_params = [
            ('up', 512, 512),   #(4,4,512)
            ('up', 512*2, 512),   #(8,8,512)
            ('up', 512*2, 512),   #(16,16,512)
            ('up', 512*2, 256),   #(32,32,256)
            ('up', 256*2, 128),   #(64,64,128)
            ('up', 128*2, 64),    #(128,128,64)
            ('up', 64*2, 64),     #(256,256,64)
            ('same', 64*2, 1),    #(256,256,1)
        ]
        super().__init__(encoder_params=encoder_params, decoder_params=decoder_params)
        
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
        
    def train_all(self):
        # if torch.cuda.is_available():
        #     self.cuda()
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
            
            #Save best model
            if best_running_loss == -1 or val_running_loss < best_running_loss:
                print(f"\nSave the best model (epoch: {epoch})\n")
                save_model(self, epoch, self.optimizer, args.save_best_model_path)
                best_running_loss = val_running_loss
        #Save last model
        print(f"\nSave the last model (epoch: {epoch})\n")
        save_model(self, epoch, self.optimizer, args.save_last_model_path)

        #Save plot
        save_plots([], [], train_loss, val_loss, args.save_plot_path)
        
    def train_epoch(self, epoch):
        self.train()
        running_loss = 0
        for step, x in enumerate(self.train_dataloader):
            # if torch.cuda.is_available():
            #     x = x.cuda()
            x = x.reshape(-1,x.shape[2],x.shape[3])
            x = x.unsqueeze(1)
            pred = self(x)
            
            self.optimizer.zero_grad()
            loss = self.criterion(pred, x)
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
            for step, x in enumerate(self.val_dataloader):
                # if torch.cuda.is_available():
                #     x = x.cuda()
                x = x.reshape(-1,x.shape[2],x.shape[3])
                x = x.unsqueeze(1)
                pred = self(x)
                
                loss = self.criterion(pred, x)      
                
                running_loss += loss.item()
                msg = f"\r| Step: {step}/{len(self.val_dataloader)} of epoch {epoch} | Val Loss: {loss:#.4} |"
                sys.stdout.write(msg)
                sys.stdout.flush()
        return running_loss / len(self.val_dataloader)
    
    def inference(self):
        #Load pretrain
        load_model(self, self.optimizer, save_file=args.save_best_model_path) + 1
        with torch.no_grad():
            rand_index = random.choice(range(len(self.val_dataloader)))
            x = self.val_dataset[rand_index]
            x = x.unsqueeze(0)
            if torch.cuda.is_available():
                x.cuda()
            pred = self(x)            
            loss = self.criterion(pred, x)      
            print(f'Loss: {loss}')
            import imageio
            result_img = torch.zeros(pred.shape[0], pred.shape[0]*2)
            result_img[:,:pred.shape[0]] = pred[0,0]
            result_img[:,pred.shape[0]:] = x
            imageio.imsave('result/fake.jpg',result_img)
            
    def extract_feature(self, x=None):
        #Load pretrain
        load_model(self, self.optimizer, save_file=args.save_best_model_path) + 1
        with torch.no_grad():
            if x is None:
                rand_index = random.choice(range(len(self.val_dataloader)))
                x = self.val_dataset[rand_index]
            x = x.unsqueeze(0)
            print(x.shape)
            path = self.val_dataset.get_item_path(rand_index)
        feature = super().extract_feature(x)
        feature = feature.cpu().detach().numpy().tolist()
        with open('result/feature.json','wt') as f:
            json.dump({'path': path, 'feature': feature}, f)
            
    def decoder(self, feature=None, sample_image=None):
        with torch.no_grad():
            #Load pretrain
            load_model(self, self.optimizer, save_file=args.save_best_model_path) + 1
            with open('result/feature.json','rt') as f:
                data = json.load(f)
                feature = data['feature']
                img_path = os.path.join('/root/Datasets/Features/M003/faces/neutral/level_1/00001/','00001.jpg')
            import imageio
            raw_image = imageio.imread(img_path)
            raw_image = raw_image / 255.0    
            raw_image = torch.from_numpy(raw_image.astype(np.float32))
            image = raw_image.unsqueeze(0).unsqueeze(0)
            feature = torch.FloatTensor(feature)
            image_result = super().decoder(feature, image)   
            
            result_img = torch.zeros(raw_image.shape[0], raw_image.shape[0]*2)
            result_img[:,:raw_image.shape[0]] = image_result[0,0]
            result_img[:,raw_image.shape[0]:] = raw_image
            imageio.imsave('result/fake1.jpg',result_img)     
        return image_result
            
if __name__ == '__main__': 
    net = GrayImage_UNet()
    # net.train_all()
    # net.inference()
    # net.extract_feature()
    net.decoder()
