import torch
from torch import nn,optim
import argparse
import numpy as np

class StackOverflow_Model(nn.Module):
    def __init__(self, vocab,sequence_length):
        super().__init__()
        self.lstm_size = 670
        self.embedding_dim = 96
        self.num_layers = 1
        self.sequence_length = sequence_length

        n_vocab = len(vocab)
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

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self):
        return (torch.zeros(self.num_layers, self.sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, self.sequence_length, self.lstm_size))

    def evaluate(self,dataloader,loss_func=nn.CrossEntropyLoss()):
        self.eval() # prep model for evaluation
        server_loss = 0
        for data, target in dataloader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = self(data)
            # calculate the loss
            loss = loss_func(output[0], target)
            # update running validation loss
            server_loss += loss.item()/data.size(0)
        server_loss = server_loss/len(dataloader)
        return server_loss

    def train_model(self, dataloader,optimizer,loss_func=nn.CrossEntropyLoss(),epochs = 1):
        self.train()
        for epoch in range(epochs):
            state_h, state_c = self.init_state()
            for batch, (x, y) in enumerate(dataloader):
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
                print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })
        return loss.item()



    def get_weights(self):
        return  copy.deepcopy(self.state_dict())

    def set_weights(self, model_state):
        self.load_state_dict(copy.deepcopy(model_state))
