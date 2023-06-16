import json
import argparse
import numpy as np
import sys
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
parser.add_argument('--train_dataset_path', type=str, default=f'data/train_{dataset}.json')
parser.add_argument('--val_dataset_path', type=str, default=f'data/val_{dataset}.json')

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=1.0e-4)
parser.add_argument('--n_epoches', type=int, default=500)

parser.add_argument('--save_path', type=str, default=f'result_{filename}_v3_{dataset}')
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
        
    def __getitem__(self, index=None, start=None, end=None):
        if index is None:
            while True:
                index = random.choice(range(len(self.data_path)))
                parts = self.data_path[index].split('|') 
                aufeat_path = parts[0].replace('mfccs', 'aufeat50')

                for i in range(int(parts[3]), int(parts[4])):
                    aufeat_subpath = os.path.join(aufeat_path, f'{i:05d}.npy')
                    if not os.path.exists(aufeat_subpath):
                        print(aufeat_subpath)
                        break
                if i == int(parts[4])-1:
                    break

        parts = self.data_path[index].split('|')   
        aufeat_path = parts[0].replace('mfccs', 'aufeat50')
             
        if start is None and end is None:
            start = parts[3]
            end = parts[4]    
        # if args.train:
        data = read_data_from_path(mfcc_path=parts[0], lm_path=parts[1], start=start, end=end)
        aufeat = read_aufeat_from_path(aufeat_path, start=start, end=end)
        # else:
        #     data = read_data_from_path(mfcc_path=parts[0], lm_path=parts[1])
        return  torch.from_numpy(data['mfcc_data_list']), torch.from_numpy(data['lm_data_list']), data['bb_list'],parts[0], torch.from_numpy(aufeat)

    def __len__(self):
        return len(self.data_path)
        # return 32
        
class A2LM(nn.Module):
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
        
        self.attention_layer = nn.Linear(in_features=self.lm_hidden_size, out_features=(self.lm_hidden_size + 64), bias=True)
        
        self.fc1 = nn.Linear(self.lm_hidden_size*2+64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.output_size)
        
        self.ae = AE()
        
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
        checkpoint = torch.load('/root/TalkingHeadProject/my_framework/result_Audio_Emotional_Embedding_MEAD/best_model.pt', map_location=f'cuda:{torch.cuda.current_device()}')
        self.ae.load_state_dict(checkpoint["model_state"])
        print(f'Load Emotion Embedding epoch: {checkpoint["epoch"]}')
        
            
    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print("Trainable Parameters: %.3fM" % parameters)
        return parameters 
    
    def forward(self, audio, aufeat):
        emo_embedding = []
        for i in range(aufeat.shape[0]):
            emo = self.ae(aufeat[i:i+1])
            emo_embedding.append(emo)
        emo_embedding = torch.stack(emo_embedding, dim=0) #(1,25,64)
        
        #audio: batch_size, seq_len, dim
        audio_out,_ = self.audio_lstm(audio)
        lm_in = torch.zeros((audio_out.shape[0], audio_out.shape[1], audio_out.shape[2]*2)).cuda()
        lm_in[:,0:1,:] = torch.cat((audio_out[:,0:1,:], audio_out[:,0:1,:]), dim=2)
        for i in range(1, audio_out.size(1)):
            lm_in[:,i:i+1,:] = torch.cat((audio_out[:,i-1:i,:], audio_out[:,i:i+1,:]), dim=2)
            
        lm_out,_ = self.lm_lstm(lm_in) #1,25,512
        lm_out = self.attention_layer(lm_out)  
        energy = torch.bmm(torch.cat((audio_out, emo_embedding), dim=2), lm_out.transpose(1, 2))
        attention_weights = F.softmax(energy, dim=2)
        context_vector = torch.bmm(attention_weights, lm_out)
        output = torch.cat([audio_out, context_vector], dim=2)
        # fc_in = output.reshape(-1, self.lm_hidden_size*2)
        fc_out = self.fc1(output)
        fc_out = self.fc2(fc_out)
        fc_out = self.fc3(fc_out)
        # lm_pred = fc_out.reshape(audio.shape[0], audio.shape[1], -1)
        return fc_out
        
        
    def train_all(self):
        if torch.cuda.is_available():
            self.cuda()
        #Load pretrain
        current_epoch = 0
        if args.use_pretrain == True:
            current_epoch = self.load_model('e250-2023-05-13 12:42:04.792907.pt') + 1
        
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
        for step, (audio, lm_gt, _,_, aufeat) in enumerate(self.train_dataloader):            
            if torch.cuda.is_available():
                audio, lm_gt, aufeat = audio.cuda(), lm_gt.cuda(), aufeat.cuda()      
                audio = audio.reshape(audio.shape[0], audio.shape[1], -1)   #1,25,28*12

            lm_pred = self(audio, aufeat)   #1,25,68*2
            
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
            for step, (audio, lm_gt, _,_,aufeat) in enumerate(self.val_dataloader):
                if torch.cuda.is_available():
                    audio, lm_gt, aufeat = audio.cuda(), lm_gt.cuda(), aufeat.cuda()      
                    audio = audio.reshape(audio.shape[0], audio.shape[1], -1)   #1,25,28*12
                    
                lm_pred = self(audio,aufeat)       #1,25,68*2
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
            # rand_index = random.choice(range(len(self.val_dataloader)))
            audio,lm_gt,lm_bb,_,aufeat = self.val_dataset.__getitem__(None, 0, -1)
            audio = audio.unsqueeze(0)    #x = 1,25,28,12; y = 1,25,68*2
            audio = audio.reshape(audio.shape[0], audio.shape[1], -1)   #1,25,28*12
            lm_gt = lm_gt.unsqueeze(0)
            aufeat = aufeat.unsqueeze(0)
            if torch.cuda.is_available():
                audio,lm_gt, aufeat = audio.cuda(), lm_gt.cuda() , aufeat.cuda()  
            
            lm_pred = self(audio, aufeat)           #1,25,68*2
            print(lm_pred.shape)
            print(lm_gt.shape)
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
                result_img = np.zeros((256, 256, 3))
                result_img[:,:,0:1] = outputs_gt[i] * 255
                result_img[:,:,1:2] = outputs_pred[i] * 255
                outputs.append(result_img)
            
            create_video(outputs,f'{args.save_path}/prediction.mp4',fps=10)
            
    def inference_each_emo(self):
        if torch.cuda.is_available():
            self.cuda()
        with torch.no_grad():
            mfcc_paths = ['/root/Datasets/CREMA-D/Features/mfccs/1001_TIE_ANG_XX',
                            '/root/Datasets/CREMA-D/Features/mfccs/1001_ITS_DIS_XX',
                            '/root/Datasets/CREMA-D/Features/mfccs/1001_ITS_FEA_XX',
                            '/root/Datasets/CREMA-D/Features/mfccs/1001_TIE_HAP_XX',
                            '/root/Datasets/CREMA-D/Features/mfccs/1001_WSI_SAD_XX',
                            '/root/Datasets/CREMA-D/Features/mfccs/1001_TIE_NEU_XX']
            for id, mfcc_path in enumerate(mfcc_paths):
                lm_path = mfcc_path.replace('mfccs','landmarks74')
                data = read_data_from_path(mfcc_path=mfcc_path, lm_path=lm_path, start=0, end=-1)
                audio = torch.from_numpy(data['mfcc_data_list'])
                lm_gt = torch.from_numpy(data['lm_data_list'])
                lm_bb = data['bb_list']
        
                audio = audio.unsqueeze(0)    #x = 1,25,28,12; y = 1,25,68*2
                audio = audio.reshape(audio.shape[0], audio.shape[1], -1)   #1,25,28*12
                lm_gt = lm_gt.unsqueeze(0)
                if torch.cuda.is_available():
                    audio,lm_gt = audio.cuda(), lm_gt.cuda()   
                
                lm_pred = self(audio)           #1,25,68*2

                lm_pred_newshape = lm_pred.reshape(lm_gt.shape[0],lm_gt.shape[1],68,2)
                lm_gt_newshape = lm_gt.reshape(lm_gt.shape[0],lm_gt.shape[1],68,2)
                lmd = calculate_LMD_torch(lm_pred_newshape, 
                                        lm_gt_newshape, 
                                        norm_distance=1)
                print(f"LMD_{id}: {lmd}") 
                
                lm_pred = lm_pred.reshape(lm_pred.size(1),68,2)
                lm_pred = lm_pred.cpu().detach().numpy()
                outputs_pred = connect_face_keypoints(256,256,lm_pred)
                
                lm_gt = lm_gt.reshape(lm_gt.size(1),68,2)
                lm_gt = lm_gt.cpu().detach().numpy()
                outputs_gt = connect_face_keypoints(256,256,lm_gt)
                
                # Save lm_pred, lm_gt
                np.save(f'{args.save_path}/lm_pred_{id}.npy', lm_pred)
                np.save(f'{args.save_path}/lm_gt_{id}.npy', lm_gt)
                np.save(f'{args.save_path}/lm_bb_{id}.npy', lm_bb)
                
                outputs = []
                for i in range(len(outputs_gt)):
                    result_img = np.zeros((256, 256*2, 1))
                    result_img[:,:256,:] = outputs_gt[i] * 255
                    result_img[:,256:,:] = outputs_pred[i] * 255
                    outputs.append(result_img)
                
                create_video(outputs,f'{args.save_path}/prediction_{id}.mp4',fps=10)
    
    def calculate_val_lmd(self):
        if torch.cuda.is_available():
            self.cuda()
        lmd_loss = 0
        lmv_loss = 0
        fld_loss = 0
        flvd_loss = 0
        rmse_loss = 0
        mae_loss = 0
        lmd_min = [100,100,100,100,100,100,100,100]
        lmd_min_path = ['','','','','','','','']
        # emo_mapping = ['angry', 'disgusted', 'contempt', 'fear', 'happy', 'sad', 'surprised', 'neutral']
        emo_mapping = ['ANG', 'DIS', 'contempt', 'FEA', 'HAP', 'SAD', 'surprised', 'NEU']

        with torch.no_grad():
            for step, (audio, lm_gt, _, data_path,  aufeat) in tqdm(enumerate(self.val_dataloader)):
                if torch.cuda.is_available():
                    audio,lm_gt, aufeat = audio.cuda(), lm_gt.cuda() , aufeat.cuda()
                audio = audio.reshape(audio.shape[0], audio.shape[1], -1)   #1,25,28*12
                lm_pred = self(audio, aufeat)       #1,25,68*2
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
                
                fld = calculate_LMD_torch(lm_pred_newshape[:,:,:48,:], 
                                        lm_gt_newshape[:,:,:48,:], 
                                        norm_distance=1)
                fld_loss += fld 
                
                flvd = calculate_LMV_torch(lm_pred_newshape[:,:,:48,:], 
                                        lm_gt_newshape[:,:,:48,:], 
                                        norm_distance=1)
                flvd_loss += flvd
                
                rmse = calculate_rmse_torch(lm_pred_newshape, lm_gt_newshape)
                rmse_loss += rmse
                
                mae = calculate_mae_torch(lm_pred_newshape, lm_gt_newshape)
                mae_loss += mae
                
                for id, emo in enumerate(emo_mapping):
                    if emo in data_path[0]:                        
                        if lmd < lmd_min[id]:
                            lmd_min[id] = lmd
                            lmd_min_path[id] = data_path[0]
                            
        for id in range(len(lmd_min)):
            print(f'{id} : {lmd_min[id]} - {lmd_min_path[id]}')
                
        return lmd_loss / len(self.val_dataloader), lmv_loss / len(self.val_dataloader),fld_loss / len(self.val_dataloader), flvd_loss / len(self.val_dataloader), rmse_loss / len(self.val_dataloader), mae_loss / len(self.val_dataloader)
    
    def load_model(self, filename='best_model.pt'):
        return load_model(self, self.optimizer, save_file=f'{args.save_path}/{filename}')
            
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 25
        self.output_emo_size = 8
        self.output_lv_size = 3
        self.fc1 = nn.Sequential(
                nn.Linear(in_features=self.input_size, out_features=512),
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
                nn.Linear(in_features=256, out_features=128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2)
                )
        
        self.fc3 = nn.Sequential(
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, self.output_emo_size)
                )
        
        self.fc4 = nn.Sequential(
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, self.output_lv_size)
                )
    
    def forward(self, audio):
        #audio: batch_size, dim
        bs, seq, dim = audio.shape
        output = audio.reshape(-1, dim)
        output = self.fc1(output)
        output = output.reshape(bs, seq, -1)
        output, _ = self.LSTM(output)
        output = output.reshape(-1, output.shape[-1])
        output = self.fc2(output)
        return output
        # output1 = self.fc3(output)
        # # output1 = output1.view(bs, seq, -1)
        # output2 = self.fc4(output)
        # # output2 = output2.view(bs, seq, -1)
        # return output1, output2
            
if __name__ == '__main__': 
    net = A2LM()
    if args.train:
        net.train_all()
    elif args.val:
        net.load_model()
        lmd, lmv, fld, flvd, rmse,mae = net.calculate_val_lmd()
        print(f'LMD: {lmd};LMV: {lmv};F-LD: {fld};F-LVD: {flvd}; RMSE: {rmse}; MAE: {mae}')
        #Epoch 50/MEAD:  LMD: 4.410379845921586;LMV: 2.264491319656372;F-LD: 4.105265893587252;F-LVD: 2.0177090167999268; RMSE: 5.0789794921875; MAE: 2.6438090801239014
        #Epoch 100/MEAD: LMD: 3.996739890517258;LMV: 2.1682395935058594;F-LD: 3.3555815685086134;F-LVD: 1.949436902999878; RMSE: 4.3910932540893555; MAE: 2.228925943374634
        #Epoch 150/MEAD: LMD: 3.1293318184410652;LMV: 1.9373419284820557;F-LD: 2.8211854579972058;F-LVD: 1.7920507192611694; RMSE: 3.734647750854492; MAE: 1.8377188444137573
        #Epoch 200/MEAD: LMD: 2.663476717181322;LMV: 1.7163478136062622;F-LD: 2.448796609552895;F-LVD: 1.6322695016860962; RMSE: 3.399491310119629; MAE: 1.5840797424316406
        #Epoch 250/MEAD: LMD: 2.4293050213557916;LMV: 1.5708444118499756;F-LD: 2.2642236802636124;F-LVD: 1.5167795419692993; RMSE: 3.2338504791259766; MAE: 1.4591704607009888
        #Epoch 300/MEAD: LMD: 2.3062494862370375;LMV: 1.4722583293914795;F-LD: 2.1150948855935074;F-LVD: 1.4372278451919556; RMSE: 3.1123502254486084; MAE: 1.369788408279419
        #Epoch 350/MEAD: LMD: 2.295786046400303;LMV: 1.4060178995132446;F-LD: 2.081645365168409;F-LVD: 1.377501130104065; RMSE: 3.0708112716674805; MAE: 1.3527467250823975
        #Epoch 400/MEAD: LMD: 2.1176823348533818;LMV: 1.3401939868927002;F-LD: 1.9752322071936073;F-LVD: 1.3183976411819458; RMSE: 2.976276397705078; MAE: 1.2722632884979248

        
        #Epoch 100/MEAD: LMD: 3.5444338350761226;LMV: 2.01357364654541; RMSE: 4.393388271331787; MAE: 2.229139566421509
        #Epoch 300/MEAD: LMD: 2.172164072350758;LMV: 1.4477765560150146; RMSE: 3.112738609313965; MAE: 1.3703311681747437
        #Epoch 484/MEAD: LMD: 1.8943945998098792;LMV: 1.245319128036499; RMSE: 2.876394033432007; MAE: 1.1945229768753052
        #Epoch 105/CRMD: LMD: 1.8502711271867156;LMV: 1.41961669921875; RMSE: 2.289553165435791; MAE: 1.1693556308746338
        #Epoch 196/CRMD: LMD: 1.5343044173593323;LMV: 1.2477023601531982; RMSE: 1.9643146991729736; MAE: 0.9698354601860046
        #Epoch 300/CRMD: LMD: 1.4413805628816287;LMV: 1.1612162590026855; RMSE: 1.8930352926254272; MAE: 0.9108084440231323
        #Epoch 385/CRMD: LMD: 1.3678209967911243;LMV: 1.1221470832824707; RMSE: 1.8277004957199097; MAE: 0.8646678328514099
    else:
        net.load_model()
        # net.inference_each_emo()
        net.inference()
