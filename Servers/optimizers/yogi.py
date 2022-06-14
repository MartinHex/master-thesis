from Servers.optimizers.server_optimizer import server_optimizer
from collections import defaultdict
import torch

class Yogi(server_optimizer):
    def __init__(self,lr=1,tau=0.0001,b1=0.9,b2=0.999):
        super().__init__()
        self.lr = lr
        self.tau = tau
        self.b1 = b1
        self.b2 = b2
        self.v = 0
        self.m = 0

    def get_grad(self,grad):
        self.m = self.b1 * self.m + (1-self.b1) * grad
        v_square = torch.square(grad)
        self.v = self.v - (1-self.b2) * v_square * torch.sign(self.v - v_square)
        grad_new = self.lr * self.m.div(torch.sqrt(self.v)+self.tau)
        return grad_new
