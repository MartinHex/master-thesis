import torch
from torch import nn
from Models.nn_Model import nn_Model

class StackOverflow_Model(nn_Model):
    def __init__(self, n_vocab=10004,sequence_length=20):
        super().__init__()
        self.lstm_size = 670
        self.embedding_dim = 96
        self.num_layers = 1
        self.sequence_length = sequence_length
        self.batch_size = 16

        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
        )
        self.fc1 = nn.Linear(self.lstm_size, self.embedding_dim)
        self.fc2 = nn.Linear(self.embedding_dim, n_vocab)
        #self.softmax = nn.Softmax(dim = -1)

    def forward(self, x, prev_state=None):
        if(prev_state==None):
            prev_state=self.init_state()
        embed = self.embedding(torch.transpose(x, 0, 1))
        output, state = self.lstm(embed)
        for batch_id in range(output.shape[1]):
            batch_pred = torch.tensor([])
            for token in range(output.shape[0]):
                temp_output = output[token,batch_id, :].flatten()
                temp_output = self.fc1(temp_output)
                temp_output = self.fc2(temp_output)
                if token == 0:
                    batch_pred = torch.reshape(temp_output, (1, len(temp_output)))
                else:
                    batch_pred = torch.cat((batch_pred, torch.reshape(temp_output, (1, len(temp_output)))), dim = 0)
            if batch_id == 0:
                output_predictions = torch.reshape(batch_pred, (1, batch_pred.shape[1], batch_pred.shape[0]))
            else:
                output_predictions = torch.cat((output_predictions, torch.reshape(batch_pred, (1, batch_pred.shape[1], batch_pred.shape[0]))), dim = 0)
        return output_predictions, state

    def predict(self, input, device = None):
        self.eval()
        state_h, state_c = self.init_state()
        if (device!= None):
            self.to(device)
            state_h = state_h.to(device)
            state_c = state_c.to(device)
            input = input.to(device)
        pred, (state_h, state_c) = self(input, (state_h, state_c))
        if device != None: pred = pred.to(device)
        if (device!= None):
            self.to('cpu')
            torch.cuda.empty_cache()
        pred = torch.reshape(pred[:,:,-1], (pred.shape[0], pred.shape[1]))
        return pred.float(), None

    def init_state(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.lstm_size),
                torch.zeros(self.num_layers, self.batch_size, self.lstm_size))

    def evaluate(self,dataloader,loss_func=nn.CrossEntropyLoss(), device = None, take_mean=True):
        self.eval() # prep model for evaluation
        loss_per_batch = []
        state_h, state_c = self.init_state()
        if (device!= None):
            self.to(device)
            state_h = state_h.to(device)
            state_c = state_c.to(device)
        for i, (x,y) in enumerate(dataloader):

            if(device!= None):
                x = x.to(device)
                y = y.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            y_pred, (state_h, state_c) = self(x, (state_h, state_c))
            if device != None: y_pred = y_pred.to(device)
            # Select last word as prediction
            for token in range(y_pred.shape[2]):
                token_pred = torch.reshape(y_pred[:,:,token], (y_pred.shape[0], y_pred.shape[1]))
                if token == 0:
                    loss = loss_func(
                        token_pred,
                        y[:,token].flatten()
                    )
                else:
                    loss = loss.add(
                        loss_func(
                            token_pred,
                            y[:,token].flatten()
                        )
                    )
            #loss = loss_func(y_pred, y)
            # update running validation loss
            loss_per_batch.append(loss.item()/x.size(0))
        if (device!= None):
            self.to('cpu')
            torch.cuda.empty_cache()
        if take_mean:
            return sum(loss_per_batch)/len(loss_per_batch)
        else:
            return loss_per_batch

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
                if device != None: y_pred = y_pred.to(device)
                for token in range(y_pred.shape[2]):
                    token_pred = torch.reshape(y_pred[:,:,token], (y_pred.shape[0], y_pred.shape[1]))
                    print(token_pred.shape)
                    if token == 0:
                        loss = loss_func(
                            token_pred,
                            y[:,token].flatten()
                        )
                    else:
                        loss = loss.add(
                            loss_func(
                                token_pred,
                                y[:,token].flatten()
                            )
                        )
                # Detatch reference for maintaining graph-structure
                state_h = state_h.detach()
                state_c = state_c.detach()
                # backpropagation, compute gradients
                loss.backward()
                # apply gradients
                optimizer.step()
        if (device!= None):
            self.to('cpu')
            torch.cuda.empty_cache()
        return loss.item()

    def iter_train_model(self, dataloader,optimizer,loss_func=nn.CrossEntropyLoss(),
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
                if device != None: y_pred = y_pred.to(device)
                for token in range(y_pred.shape[2]):
                    token_pred = torch.reshape(y_pred[:,:,token], (y_pred.shape[0], y_pred.shape[1]))
                    print(token_pred.shape)
                    if token == 0:
                        loss = loss_func(
                            token_pred,
                            y[:,token].flatten()
                        )
                    else:
                        loss = loss.add(
                            loss_func(
                                token_pred,
                                y[:,token].flatten()
                            )
                        )
                # Detatch reference for maintaining graph-structure
                state_h = state_h.detach()
                state_c = state_c.detach()
                # backpropagation, compute gradients
                loss.backward()
                # apply gradients
                optimizer.step()
                yield self.get_weights()
        if (device!= None):
            self.to('cpu')
            torch.cuda.empty_cache()
        return loss.item()
