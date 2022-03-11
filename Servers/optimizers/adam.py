from Servers.optimizers.server_optimizer import server_optimizer
from collections import defaultdict
import torch

class Adam(server_optimizer):
    def __init__(self,lr=1,tau=0.0001,b1=0.9,b2=0.999):
        super().__init__()
        self.lr = lr
        self.tau = tau
        self.b1 = b1
        self.b2 = b2
        self.v = defaultdict(lambda:0)
        self.m = defaultdict(lambda:0)

    def get_grad(self,grad):
        grad_new = {}
        for key in grad:
            self.m[key] = self.b1*self.m[key]+(1-self.b1)*grad[key]
            self.v[key] = self.v[key]+(1-self.b2)*torch.square(grad[key])
            grad_new[key] = self.lr*self.m[key].div(torch.sqrt(self.v[key])+self.tau)
        return grad_new
