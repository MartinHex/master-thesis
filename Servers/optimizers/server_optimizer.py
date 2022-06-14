from abc import ABC,abstractmethod

class server_optimizer(ABC):

    def opt(self,w_new,w_old):
        grad = w_new.sub(w_old)
        new_grad = self.get_grad(grad)
        w =  w_old.add(new_grad)
        return w

    @abstractmethod
    def get_grad(self,grad):
        pass
