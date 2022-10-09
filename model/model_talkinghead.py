from gc import isenabled
from torch import nn
import numpy as np
import torch
import os
from os.path import join
import matplotlib.pyplot as plt

class Model_TalkingHead(nn.Module):
    def __init__(self):
        super(Model_TalkingHead, self).__init__()
        self.config = None
        self.optim = None
        self.loss_fn = None
        self.register_buffer("step", torch.zeros(1, dtype=torch.long))
        self.best_valid_loss = float('inf')

    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def update(self, y_pred, y):
        loss = self.loss_fn(y_pred, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss

    def calculate_loss(self, y_pred, y):
        return self.loss_fn(y_pred, y)

    def get_step(self):
        return self.step.data.item()

    def reset_step(self):
        self.step = self.step.data.new_tensor(1)

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print("Trainable Parameters: %.3fM" % parameters)
        return parameters

    def load(self, load_optim = True):
        if next(self.parameters()).is_cuda and torch.cuda.is_available():
            checkpoint = torch.load(str(self.config['pretrain_path']), map_location=f'cuda:{torch.cuda.current_device()}')
        else:
            checkpoint = torch.load(str(self.config['pretrain_path']), map_location='cpu')
        self.load_state_dict(checkpoint["model_state"])

        if load_optim == True and "optimizer_state" in checkpoint:
            self.optim.load_state_dict(checkpoint["optimizer_state"])

        fname = os.path.basename(self.config['pretrain_path'])
        epoch = checkpoint["epoch"]
        print(f"Load pretrained model {fname} | Epoch: {epoch}")
        return epoch

    def save(self, epoch, loss, save_optim = True):
        os.makedirs(join(self.config['save_path'],'backups'), exist_ok=True)
        save_file = join(self.config['save_path'], 'backups' , f'epoch_{epoch:06d}.pt')
        if save_optim:
            torch.save({
                "epoch": epoch,
                "model_state": self.state_dict(),
                "loss": loss,
                "optimizer_state": self.optim.state_dict(),
            }, str(save_file))
        else:
            torch.save({
                "epoch": epoch,
                "model_state": self.state_dict(),
                "loss": loss,
            }, str(save_file))

    def save_best_model(self, epoch, loss, save_optim = True):
        if loss < self.best_valid_loss:
            self.best_valid_loss = loss
            os.makedirs(self.config['save_path'], exist_ok=True)
            save_file = join(self.config['save_path'], f'best_model.pt')
            if save_optim:
                torch.save({
                    "epoch": epoch,
                    "model_state": self.state_dict(),
                    "loss": loss,
                    "optimizer_state": self.optim.state_dict(),
                }, str(save_file))
            else:
                torch.save({
                    "epoch": epoch,
                    "model_state": self.state_dict(),
                    "loss": loss,
                }, str(save_file))

    def save_plots(self, train_acc, valid_acc, train_loss, valid_loss):
        """
        Function to save the loss and accuracy plots to disk.
        """
        # accuracy plots
        plt.figure(figsize=(10, 7))
        plt.plot(
            train_acc, color='green', linestyle='-', 
            label='train accuracy'
        )
        plt.plot(
            valid_acc, color='blue', linestyle='-', 
            label='validataion accuracy'
        )
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(join(self.config['save_path'],'accuracy.png'))
        
        # loss plots
        plt.figure(figsize=(10, 7))
        plt.plot(
            train_loss, color='orange', linestyle='-', 
            label='train loss'
        )
        plt.plot(
            valid_loss, color='red', linestyle='-', 
            label='validataion loss'
        )
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(join(self.config['save_path'],'loss.png'))