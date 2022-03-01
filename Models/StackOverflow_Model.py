import torch
from torch import nn,optim
import argparse
import numpy as np
import copy

class StackOverflow_Model(nn.Module):
    def __init__(self, n_vocab=10004,sequence_length=20):
        super().__init__()
        self.lstm_size = 670
        self.embedding_dim = 96
        self.num_layers = 1
        self.sequence_length = sequence_length

        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state=None):
        if(prev_state==None):
            prev_state=self.init_state()
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        p = nn.Softmax(logits).dim
        return p, state

    def predict(self, input, device = None):
        self.eval()
        state_h, state_c = self.init_state()
        if (device!= None):
            self.to(device)
            state_h = state_h.to(device)
            state_c = state_c.to(device)
            input = input.to(device)
        return self(input, (state_h, state_c))

    def init_state(self):
        return (torch.zeros(self.num_layers, self.sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, self.sequence_length, self.lstm_size))

    def evaluate(self,dataloader,loss_func=nn.CrossEntropyLoss(), device = None):
        self.eval() # prep model for evaluation
        avg_loss = 0
        state_h, state_c = self.init_state()
        if (device!= None):
            self.to(device)
            state_h = state_h.to(device)
            state_c = state_c.to(device)
        for x,y in dataloader:
            if(device!= None):
                x = x.to(device)
                y = y.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            y_pred, (state_h, state_c) = self(x, (state_h, state_c))
            loss = loss_func(y_pred.transpose(1, 2), y)
            # update running validation loss
            avg_loss += loss.item()/x.size(0)
        avg_loss = avg_loss/len(dataloader)
        return avg_loss

    def train_model(self, dataloader,optimizer,loss_func=nn.CrossEntropyLoss(),
                    epochs = 1,device=None):
        if (device!= None): self.to(device)
        self.train()
        for epoch in range(epochs):
            state_h, state_c = self.init_state()
            if(device!= None):
                state_h = state_h.to(device)
                state_c = state_c.to(device)
            for batch, (x, y) in enumerate(dataloader):
                if(device!= None):
                    x = x.to(device)
                    y = y.to(device)
                # gives batch data, normalize x when iterate train_loader
                y_pred, (state_h, state_c) = self(x, (state_h, state_c))
                loss = loss_func(y_pred.transpose(1, 2), y)
                # Detatch reference for maintaining graph-structure
                state_h = state_h.detach()
                state_c = state_c.detach()
                # backpropagation, compute gradients
                loss.backward()
                # apply gradients
                optimizer.step()
        return loss.item()



    def get_weights(self):
        return  copy.deepcopy(self.state_dict())

    def set_weights(self, model_state):
        self.load_state_dict(copy.deepcopy(model_state))
