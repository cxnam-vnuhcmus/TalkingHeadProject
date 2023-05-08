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
from evaluation.evaluation_landmark import *

parser = argparse.ArgumentParser()
parser.add_argument('--train_dataset_path', type=str, default='data/train_MEAD.json')
parser.add_argument('--val_dataset_path', type=str, default='data/val_MEAD.json')

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=1.0e-4)
parser.add_argument('--n_epoches', type=int, default=500)

parser.add_argument('--save_path', type=str, default='result_A2LM_LMAudioPrev_v2')
parser.add_argument('--use_pretrain', type=bool, default=False)
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
        rand_index = random.choice(range(len(self.data_path)))
        parts = self.data_path[rand_index].split('|')        

        # if args.train:
        data = read_data_from_path(mfcc_path=parts[0], lm_path=parts[1], start=parts[3], end=parts[4])
        # else:
        #     data = read_data_from_path(mfcc_path=parts[0], lm_path=parts[1])
        return  torch.from_numpy(data['mfcc_data_list']), torch.from_numpy(data['lm_data_list']), data['bb_list']

    def __len__(self):
        return len(self.data_path)
        # return 64
        
class A2LM_LMAudioPrev(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 28*12
        self.lm_hidden_size = 512
        self.lm_num_layers = 3
        self.audio_hidden_size = 512
        self.audio_num_layers = 3
        self.output_size = 68*2
        
        self.audio_lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.audio_hidden_size,
                            num_layers=self.audio_num_layers,
                            dropout=0,
                            bidirectional=False,
                            batch_first=True)
        
        self.lm_lstm = nn.LSTM(input_size=self.audio_hidden_size * 2,
                            hidden_size=self.lm_hidden_size,
                            num_layers=self.lm_num_layers,
                            dropout=0,
                            bidirectional=False,
                            batch_first=True)
        
        self.fc1 = nn.Linear(self.lm_hidden_size, 256)
        self.fc2 = nn.Linear(256, self.output_size)
        
        self.train_dataset = FaceDataset(args.train_dataset_path)
        self.val_dataset = FaceDataset(args.val_dataset_path)
        
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        
        self.mseloss = nn.MSELoss()
        # self.ctcloss = nn.CTCLoss(blank=0)
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
        #audio: batch_size, seq_len, dim
        audio_out,_ = self.audio_lstm(audio)
        lm_in = torch.zeros((audio_out.shape[0], audio_out.shape[1], audio_out.shape[2]*2)).cuda()
        lm_in[:,0:1,:] = torch.cat((audio_out[:,0:1,:], audio_out[:,0:1,:]), dim=2)
        for i in range(1, audio_out.size(1)):
            lm_in[:,i:i+1,:] = torch.cat((audio_out[:,i-1:i,:], audio_out[:,i:i+1,:]), dim=2)
            
        lm_out,_ = self.lm_lstm(lm_in)
        
        fc_in = lm_out.reshape(-1, self.lm_hidden_size)
        fc_out = self.fc1(fc_in)
        fc_out = self.fc2(fc_out)
        lm_pred = fc_out.reshape(audio.shape[0], audio.shape[1], -1)
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
                self.inference()
            
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
        for step, (audio, lm_gt, _) in enumerate(self.train_dataloader):            
            if torch.cuda.is_available():
                audio, lm_gt = audio.cuda(), lm_gt.cuda()      #audio = 1,25,28,12; lm = 1,25,68*2
                audio = audio.reshape(audio.shape[0], audio.shape[1], -1)   #1,25,28*12

            lm_pred = self(audio)   #1,25,68*2
            
            self.optimizer.zero_grad()
            mseloss = self.mseloss(lm_pred, lm_gt)
            # ctcloss = self.ctcloss(lm_pred, lm_gt)
            lm_pred_newshape = lm_pred.reshape(lm_gt.shape[0],lm_gt.shape[1],68,2)
            lm_gt_newshape = lm_gt.reshape(lm_gt.shape[0],lm_gt.shape[1],68,2)
            lmdloss = calculate_LMD_torch(lm_pred_newshape, 
                                    lm_gt_newshape, 
                                    norm_distance=1)
            loss = mseloss * 0.1 + lmdloss
            loss.backward()
            self.optimizer.step()
        
            running_loss += loss.item()
            msg = f"\r| Step: {step}/{len(self.train_dataloader)} of epoch {epoch} | MSE Loss {mseloss:#.4}; LMD Loss: {lmdloss:#.4}; Train Loss: {loss:#.4} |"
            sys.stdout.write(msg)
            sys.stdout.flush()
            
        return running_loss / len(self.train_dataloader)

    def validate_epoch(self, epoch):
        self.eval()
        running_loss = 0
        with torch.no_grad():
            for step, (audio, lm_gt, _) in enumerate(self.val_dataloader):
                if torch.cuda.is_available():
                    audio, lm_gt = audio.cuda(), lm_gt.cuda()      #audio = 1,25,28,12; lm = 1,25,68*2
                    audio = audio.reshape(audio.shape[0], audio.shape[1], -1)   #1,25,28*12
                
                lm_pred = self(audio)       #1,25,68*2
                mseloss = self.mseloss(lm_pred, lm_gt)
                lm_pred_newshape = lm_pred.reshape(lm_gt.shape[0],lm_gt.shape[1],68,2)
                lm_gt_newshape = lm_gt.reshape(lm_gt.shape[0],lm_gt.shape[1],68,2)
                lmdloss = calculate_LMD_torch(lm_pred_newshape, 
                                        lm_gt_newshape, 
                                        norm_distance=1)
                loss = mseloss * 0.1 + lmdloss     
                
                running_loss += loss.item()
                msg = f"\r| Step: {step}/{len(self.val_dataloader)} of epoch {epoch} | MSE Loss {mseloss:#.4}; LMD Loss: {lmdloss:#.4}; Val Loss: {loss:#.4} |"
                sys.stdout.write(msg)
                sys.stdout.flush()
        return running_loss / len(self.val_dataloader)
    
    def inference(self):
        if torch.cuda.is_available():
            self.cuda()
        with torch.no_grad():
            rand_index = random.choice(range(len(self.val_dataloader)))
            audio,lm_gt,lm_bb = self.val_dataset[rand_index]      
            audio = audio.unsqueeze(0)    #x = 1,25,28,12; y = 1,25,68*2
            audio = audio.reshape(audio.shape[0], audio.shape[1], -1)   #1,25,28*12
            lm_gt = lm_gt.unsqueeze(0)
            if torch.cuda.is_available():
                audio,lm_gt = audio.cuda(), lm_gt.cuda()   
            
            lm_pred = self(audio)           #1,25,68*2
            mseloss = self.mseloss(lm_pred, lm_gt)
            lm_pred_newshape = lm_pred.reshape(lm_gt.shape[0],lm_gt.shape[1],68,2)
            lm_gt_newshape = lm_gt.reshape(lm_gt.shape[0],lm_gt.shape[1],68,2)
            lmdloss = calculate_LMD_torch(lm_pred_newshape, 
                                    lm_gt_newshape, 
                                    norm_distance=1)    
            loss = mseloss * 0.1 + lmdloss  
            print(f'MSE Loss: {mseloss:#.4}; LMD Loss: {lmdloss:#.4}; Infer Loss: {loss:#.4}')
            
            lm_pred = lm_pred.reshape(lm_pred.size(1),68,2)
            lm_pred = lm_pred.cpu().detach().numpy()
            outputs_pred = connect_face_keypoints(256,256,lm_pred)
            
            lm_gt = lm_gt.reshape(lm_gt.size(1),68,2)
            lm_gt = lm_gt.cpu().detach().numpy()
            outputs_gt = connect_face_keypoints(256,256,lm_gt)
            
            # Save lm_pred, lm_gt
            np.save(f'{args.save_path}/lm_pred.npy', lm_pred)
            np.save(f'{args.save_path}/lm_gt.npy', lm_gt)
            np.save(f'{args.save_path}/lm_bb.npy', lm_bb)
            
            outputs = []
            for i in range(len(outputs_gt)):
                result_img = np.zeros((256, 256*2, 1))
                result_img[:,:256,:] = outputs_gt[i] * 255
                result_img[:,256:,:] = outputs_pred[i] * 255
                outputs.append(result_img)
            
            create_video(outputs,f'{args.save_path}/prediction.mp4',fps=10)
    
    def calculate_val_lmd(self):
        if torch.cuda.is_available():
            self.cuda()
        lmd_loss = 0
        lmv_loss = 0
        rmse_loss = 0
        mae_loss = 0
        with torch.no_grad():
            for step, (audio, lm_gt, _) in tqdm(enumerate(self.val_dataloader)):
                if torch.cuda.is_available():
                    audio,lm_gt = audio.cuda(), lm_gt.cuda() 
                audio = audio.reshape(audio.shape[0], audio.shape[1], -1)   #1,25,28*12
                lm_pred = self(audio)       #1,25,68*2
                lm_pred_newshape = lm_pred.reshape(lm_gt.shape[0],lm_gt.shape[1],68,2)
                lm_gt_newshape = lm_gt.reshape(lm_gt.shape[0],lm_gt.shape[1],68,2)
                
                lmd = calculate_LMD_torch(lm_pred_newshape[:,:,48:,:], 
                                        lm_gt_newshape[:,:,48:,:], 
                                        norm_distance=1)
                lmd_loss += lmd 
                
                lmv = calculate_LMV_torch(lm_pred_newshape[:,:,48:,:], 
                                        lm_gt_newshape[:,:,48:,:], 
                                        norm_distance=1)
                lmv_loss += lmv
                
                rmse = calculate_rmse_torch(lm_pred_newshape, lm_gt_newshape)
                rmse_loss += rmse
                
                mae = calculate_mae_torch(lm_pred_newshape, lm_gt_newshape)
                mae_loss += mae
                
        return lmd_loss / len(self.val_dataloader), lmv_loss / len(self.val_dataloader), rmse_loss / len(self.val_dataloader), mae_loss / len(self.val_dataloader)
    
    def load_model(self, filename='best_model.pt'):
        return load_model(self, self.optimizer, save_file=f'{args.save_path}/{filename}')
            
            
if __name__ == '__main__': 
    net = A2LM_LMAudioPrev()
    if args.train:
        net.train_all()
    elif args.val:
        net.load_model('e50-2023-04-07 20:03:14.594352.pt')
        lmd,lmv, rmse,mae = net.calculate_val_lmd()
        print(f'LMD: {lmd};LMV: {lmv}; RMSE: {rmse}; MAE: {mae}')
        #Face
        #Epoch 50/MEAD:  F-LD: 6.832689035229567;F-LVD: 3.4844164848327637; RMSE: 10.964693069458008; MAE: 4.57155179977417
        #Epoch 100/MEAD: F-LD: 5.932588269070881;F-LVD: 2.756976366043091; RMSE: 8.227020263671875; MAE: 3.9305410385131836
        #Epoch 150/MEAD: F-LD: 3.173133588418728;F-LVD: 1.9586735963821411; RMSE: 4.16469669342041; MAE: 2.097933292388916
        #Epoch 200/MEAD: F-LD: 2.4917491354593415;F-LVD: 1.6462697982788086; RMSE: 3.500657081604004; MAE: 1.602452278137207
        #Epoch 250/MEAD: F-LD: 2.2817777686002776;F-LVD: 1.504069447517395; RMSE: 3.343001127243042; MAE: 1.4679187536239624
        #Epoch 300/MEAD: F-LD: 2.137982948524196;F-LVD: 1.4205504655838013; RMSE: 3.2616524696350098; MAE: 1.372382640838623
        #Epoch 350/MEAD: F-LD: 2.0945796733949242;F-LVD: 1.3654778003692627; RMSE: 3.223966598510742; MAE: 1.3398289680480957
        #Epoch 400/MEAD: F-LD: 2.007885339783459;F-LVD: 1.3190276622772217; RMSE: 3.1622045040130615; MAE: 1.2883893251419067
        #Mouth
        #Epoch 50/MEAD:  LMD: 7.827761603564751;LMV: 3.9937427043914795; RMSE: 10.973363876342773; MAE: 4.574895858764648
        #Epoch 100/MEAD: LMD: 6.90499231873489;LMV: 3.177089214324951; RMSE: 8.249549865722656; MAE: 3.947064161300659
        #Epoch 150/MEAD: LMD: 3.7055727708630446;LMV: 2.1686954498291016; RMSE: 4.15667200088501; MAE: 2.094221830368042
        #Epoch 200/MEAD: LMD: 2.635167534758405;LMV: 1.7357745170593262; RMSE: 3.4499168395996094; MAE: 1.5956162214279175
        #Epoch 250/MEAD: LMD: 2.423348028485368;LMV: 1.5628397464752197; RMSE: 3.3498404026031494; MAE: 1.469605565071106
        #Epoch 300/MEAD: LMD: 2.2355713190102;LMV: 1.4511367082595825; RMSE: 3.232121706008911; MAE: 1.3603137731552124
        #Epoch 350/MEAD: LMD: 2.1947758633915972;LMV: 1.426905870437622; RMSE: 3.230516195297241; MAE: 1.3332767486572266
        #Epoch 400/MEAD: LMD: 2.1245031414962394;LMV: 1.381556749343872; RMSE: 3.1582114696502686; MAE: 1.290356159210205
        
        #Epoch 492/MEAD: LMD: 2.0290853104940276;LMV: 1.307906150817871; RMSE: 3.1999154090881348; MAE: 1.2804210186004639
        
    else:
        net.load_model()
        net.inference()
