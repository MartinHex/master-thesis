from Servers.FedAvgServer import FedAvgServer
from Clients.FedPaClient import FedPaClient
from Clients.SGDClient import SGDClient
from Algorithms.Algorithm import Algorithm
from datetime import datetime
import copy
from tqdm import tqdm

class FedPa(Algorithm):
    def __init__(self,dataloader,Model,
                callbacks=None,
                save_callbacks = False,
                batch_size=16,
                clients_per_round=None,
                client_lr = 0.01,
                burn_in =  0,
                K = 1,
                shrinkage = 1,
                mcmc_samples = 1,
                momentum=0,
                decay=0,
                dampening=0,
                server_optimizer='none',
                server_lr=1,
                tau=0.1,
                b1=.9,
                b2=0.99,
                server_momentum=1,
                burnin=0):

        client_dataloaders = dataloader.get_training_dataloaders(batch_size)

        self.fedPa_client = FedPaClient(Model(), None,
                        learning_rate = client_lr,
                        burn_in =  burn_in,
                        K = K,
                        shrinkage = shrinkage,
                        mcmc_samples = mcmc_samples,
                        momentum=momentum,
                        decay=momentum,
                        dampening=dampening)

        self.SGD_client = SGDClient(Model(), None,
                        learning_rate = client_lr,
                        momentum=momentum,
                        decay=momentum,
                        dampening=dampening)

        server = FedAvgServer(Model(),
                            optimizer=server_optimizer,
                            lr=server_lr,
                            tau=tau,
                            b1=b1,
                            b2=b2,
                            momentum=server_momentum)

        if(burnin<0 and not isinstance(burnin, int)):
            raise Exception('Invalid value of burnin.')
        self.burnin=burnin
        self.tot_rounds = 0

        super().__init__(server, self.SGD_client, client_dataloaders, callbacks, save_callbacks,clients_per_round=clients_per_round)

    def run(self,iterations, epochs = 1, device = None,option = 'mle',n_workers=3):
        if(option not in ['mle','single_sample','multi_sample']):
            raise Exception("""Incorrect option provided must be either 'mle', 'single_sample' or 'multi_sample'""")

        self.start_time = datetime.now()
        #self.server.push_weights(self.clients)
        for round in range(iterations):
            if(self.tot_rounds==self.burnin):
                self.clients = [copy.deepcopy(self.fedPa_client) for i in range(self.clients_per_round)]
            if(self.burnin>iterations):
                raise Exception('Burnin larger than number of iterations')

            print('---------------- Round {} ----------------'.format(round + 1))
            self.callback_data['timestamps'].append(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

            if option == 'single_sample': self.server.set_weights(self.server.sample_model())
            dataloader_sample = self.sample_dataloaders()

            for i, dataloader in enumerate(tqdm(dataloader_sample)):
                # Initialize Client to run
                self.clients[i].dataloader = dataloader
                self._set_model_weight(i, option)
                loss = self.clients[i].train(epochs = epochs, device = device)

            self.server.aggregate(self.clients, device=device)
            if (self.callbacks != None): self._run_callbacks()
            self.tot_rounds +=1

        if self.save_callbacks: self._save_callbacks()
        return None

    def reset_burnin(self):
        self.tot_rounds = 0
        self.clients = [copy.deepcopy(self.fedPa_client) for i in range(self.clients_per_round)]
