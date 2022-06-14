from Servers.optimizers.server_optimizer import server_optimizer
from collections import defaultdict

class SGD(server_optimizer):
    def __init__(self,lr=1,momentum=0,tau=0):
        super().__init__()
        self.tau = tau
        self.momentum = momentum
        self.b_t = 0
        self.lr = lr

    def get_grad(self,grad):
        grad_new= self.momentum*self.b_t+(1-self.tau)*grad
        self.b_t = grad_new.detach().clone()
        grad_new=self.lr*grad_new
        return grad_new
