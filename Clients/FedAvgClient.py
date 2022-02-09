from Clients.Base_Client import Base_Client
from torch import optim

class FedAvgClient(Base_Client):
    def __init__(self, model, dataloader, learning_rate = 0.01):
        super(model)
        self.optimizer = optim.SGD(self.model.parameters(), lr = learning_rate)
        self.dataloader = dataloader

    def train(self, epochs = 1, loss_function = nn.CrossEntropyLoss()):
        #def train(num_epochs, model, loader,optimizer,loss_func):
        self.model.train()
        for epoch in range(num_epochs):
            for i, (input_data, labels) in enumerate(self.dataloader):
                # gives batch data, normalize x when iterate train_loader
                b_x = Variable(input_data)   # batch x
                b_y = Variable(labels)   # batch y
                output = self.model(b_x)[0]
                loss = loss_function(output, b_y)
                # clear gradients for this training step
                self.optimizer.zero_grad()
                # backpropagation, compute gradients
                loss.backward()
                # apply gradients
                self.optimizer.step()

        return loss.item()
