from Servers.optimizers.server_optimizer import server_optimizer
from collections import defaultdict

class SGD(server_optimizer):
    def __init__(self,lr=1,momentum=0,tau=0):
        super().__init__()
        self.tau = tau
        self.momentum = momentum
        self.b_t = defaultdict(lambda: 0)
        self.lr = lr

    def get_grad(self,grad):
        grad_new = {}
        for key in grad:
            grad_new[key]= self.momentum*self.b_t[key]+(1-self.tau)*grad[key]
            self.b_t[key] = grad_new[key].detach().clone()
            grad_new[key]=self.lr*grad_new[key]
        return grad_new
