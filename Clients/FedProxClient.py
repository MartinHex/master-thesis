from Clients.SGDClient import SGDClient
import torch
from torch.nn import CrossEntropyLoss
import copy
class FedProxClient(SGDClient):
    def __init__(self, model, dataloader, learning_rate = 0.01,momentum=0,decay=0,dampening=0, mu = 0):
        super(FedProxClient, self).__init__(
            model,
            dataloader,
            learning_rate = learning_rate,
            momentum = momentum,
            decay = decay,
            dampening = dampening
            )
        self.mu = mu
        self.global_model = copy.deepcopy(model)

    def train(self, epochs = 1, device=None):
        self.global_model.set_weights(self.model.get_weights())
        self.model.train_model(
            self.dataloader,
            self.optimizer,
            epochs = epochs,
            device=device,
            loss_func = self._prox_loss, #lambda output, target: self._prox_loss(output, target),
            )


    def _prox_loss(self, output, target):
        cce_loss = CrossEntropyLoss()
        loss = cce_loss(output, target)
        device = output.get_device()
        global_model = self.global_model.to(device)
        current_model = self.model.to(device)
        for w_0, w_t in zip(global_model.parameters(), current_model.parameters()):
            w_0.requires_grad = False
            w_t.requires_grad = True
            prox_term = (self.mu / 2) * (w_0.to(device).sub(w_t.to(device))).norm(2)
            loss.add_(prox_term)
        return loss
