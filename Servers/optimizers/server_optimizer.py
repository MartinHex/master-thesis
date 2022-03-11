from abc import ABC,abstractmethod

class server_optimizer(ABC):

    def opt(self,w_new,w_old):
        grad = {}
        for k in w_old:
            grad[k] = w_new[k].sub(w_old[k])
        grad = self.get_grad(grad)
        w = {}
        for k in w_old:
            w[k] = w_old[k].add(grad[k])
        return w

    @abstractmethod
    def get_grad(self,grad):
        pass
