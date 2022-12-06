import argparse
from modules.train_module import *
from modules.net_module import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='MeadDataset')
parser.add_argument('--train_dataset_path', type=str, default='data/train_MEAD.json')
parser.add_argument('--val_dataset_path', type=str, default='data/val_MEAD.json')

parser.add_argument('--model_name', type=str, default='AG_net')
parser.add_argument('--loss', type=str, default='ssim')
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=1.0e-4)
parser.add_argument('--n_epoches', type=int, default=100)

parser.add_argument('--save_best_model_path', type=str, default='result/best_model.pt')
parser.add_argument('--save_last_model_path', type=str, default='result/last_model.pt')
parser.add_argument('--save_plot_path', type=str, default='result')
parser.add_argument('--use_pretrain', type=bool, default=False)
args = parser.parse_args()

if __name__ == '__main__': 
    train_dataset = get_dataset(args.dataset_name, args.train_dataset_path)
    val_dataset = get_dataset(args.dataset_name, args.val_dataset_path)
    
    train_dataloader = get_dataloader(train_dataset, batch_size=args.batch_size)
    val_dataloader = get_dataloader(val_dataset, batch_size=args.batch_size)
    
    model = get_model(args.model_name)
    if args.loss.lower() == 'ssim':
        from skimage.metrics import structural_similarity as ssim
        criterion = ssim
    else:
        criterion = globals()[args.loss]()    
    optimizer = globals()[args.optimizer](model.parameters(), lr = args.learning_rate)

    #Load pretrain
    current_epoch = 0
    if args.use_pretrain == True:
        current_epoch = load_model(model, optimizer) + 1

    train_loss = []
    val_loss = []
    best_running_loss = -1
    for epoch in range(current_epoch, args.n_epoches):
        print(f'\nTrain epoch {epoch}:\n')
        train_running_loss = train(model, train_dataloader, optimizer, criterion, epoch, **{'channel_axis':1, 'multichannel':True})
        train_loss.append(train_running_loss)
        
        print(f'\nValidate epoch {epoch}:\n')
        val_running_loss = validate(model, val_dataloader, criterion, epoch, **{'channel_axis':1, 'multichannel':True})
        val_loss.append(val_running_loss)
        
        msg = f"\n| Epoch: {epoch}/{args.n_epoches} | Train Loss: {train_running_loss:#.4} | Val Loss: {val_running_loss:#.4}"
        print(msg)
        
        #Save best model
        if best_running_loss == -1 or val_running_loss < best_running_loss:
            print(f"\nSave the best model (epoch: {epoch})\n")
            save_model(model, epoch, optimizer, args.save_best_model_path)
            best_running_loss = val_running_loss
        
    #Save last model
    print(f"\nSave the last model (epoch: {epoch})\n")
    save_model(model, epoch, optimizer, args.save_last_model_path)

    #Save plot
    save_plots([], [], train_loss, val_loss, args.save_plot_path)