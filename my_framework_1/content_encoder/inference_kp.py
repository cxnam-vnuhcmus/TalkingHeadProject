import json
import argparse
from modules.train_module import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='MeadKeypointDataset')
parser.add_argument('--mfcc_files', type=str, default='/root/Datasets/Features/M003/mfccs/neutral/level_1/00001')
parser.add_argument('--lm_files', type=str, default='/root/Datasets/Features/M003/landmarks74/neutral/level_1/00001')

parser.add_argument('--model_name', type=str, default='ContentEncoder')
parser.add_argument('--model_path', type=str, default='result_kp/best_model.pt')

parser.add_argument('--output_path', type=str, default='result_kp/predict.json')
args = parser.parse_args()

if __name__ == '__main__':
    model = get_model(args.model_name, input_ndim=28*12, output_ndim=68*2+6*2)
    load_model(model, save_file=args.model_path)

    dataset = globals()[args.dataset_name]()
    mfcc_data_list, lm_data_list, bb_list = dataset.read_data_from_path(args.mfcc_files, args.lm_files)
    
    x,y = torch.from_numpy(mfcc_data_list).unsqueeze(0), torch.from_numpy(lm_data_list).unsqueeze(0)
    if torch.cuda.is_available():
        x,y = x.cuda(), y.cuda()
    y = y.reshape(*y.shape[0:2], -1)

    pred = model(x)
        
    pred = pred.squeeze(0).to(torch.int16)
    y = y.squeeze(0).to(torch.int16)
    output_data =   {
                        "pred": pred.reshape(*pred.shape[0:-1], 68+6, 2).tolist(),
                        "gt": y.reshape(*y.shape[0:-1], 68+6, 2).tolist(),
                        "bb": bb_list
                    }
   
    print(f'Save predict landmark to: {args.output_path}')
    with open(args.output_path, 'w') as f:
        json.dump(output_data, f)
    