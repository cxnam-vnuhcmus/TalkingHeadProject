from torch import nn
import numpy as np

class Model_TalkingHead(nn.Module):
    def __init__(self):
        super(Model_TalkingHead, self).__init__()
        self.config = None
        self.optim = None
        self.loss = None

    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def update(self, y_pred, y):
        loss = self.loss(y_pred, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    # def get_step(self):
    #     return self.step.data.item()

    # def reset_step(self):
    #     # assignment to parameters or buffers is overloaded, updates internal dict entry
    #     self.step = self.step.data.new_tensor(1)

    # def log(self, path, msg):
    #     with open(path, "a") as f:
    #         print(msg, file=f)

    # def load(self, path, optimizer=None):
    #     if torch.cuda.is_available():
    #         checkpoint = torch.load(str(path)).cuda()
    #     else:
    #         checkpoint = torch.load(str(path), map_location="cpu")
        
    #     self.step = checkpoint["step"] - 1
        
    #     self.load_state_dict(checkpoint["model_state"])

    #     if "optimizer_state" in checkpoint and optimizer is not None:
    #         optimizer.load_state_dict(checkpoint["optimizer_state"])
            
    #     fname = os.path.basename(path)
    #     step = self.step.cpu().numpy()[0]
    #     print(f"Load pretrained model {fname} | Step: {step}")

    # def save(self, path, optimizer=None):
    #     if optimizer is not None:
    #         torch.save({
    #             "step": self.step + 1,
    #             "model_state": self.state_dict(),
    #             "optimizer_state": optimizer.state_dict(),
    #         }, str(path))
    #     else:
    #         torch.save({
    #             "step": self.step + 1,
    #             "model_state": self.state_dict(),
    #         }, str(path))

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print("Trainable Parameters: %.3fM" % parameters)
        return parameters
