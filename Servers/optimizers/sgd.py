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
        if(self.b_t==None):
            self.b_t = grad
        for key in grad:
            grad[key]= self.momentum*self.b_t[key]+(1-self.tau)*grad[key]
            self.b_t[key] = grad[key].clone().detach()
            grad[key]=self.lr*grad[key]
        return grad
