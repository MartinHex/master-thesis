# Algorithms
A folder of different algorithms used in the thesis.
Being inspired by the [FedOpt](https://www.mdpi.com/2076-3417/10/8/2864) framework an algorithm consists of a client optimization and a server optimization unit.
Each algorithm is henceforth a combination of ´clients` and a `server`.
All algorithms have cuda support through pytorch.

### Design
All algorithm define their `clients`, `client_generator` functionss and plug these into the `Algorithm.py` superclass.
Doing so the different algorithms just define a series of servers and way to get clients based on server-specific hyperparameters.
This way all algorithms are easily to iterate through and for custom algorithms one may use

```
clients = [Clients() for _ in range(n_clints)]
server = Server()
dataloader = Dataloader()
alg = Algorithm(server,dataloader,clients=clients)
```

or

```
clients_generator = def c_generator(): return Clients()
server = Server()
dataloader = Dataloader()
alg = Algorithm(server,dataloader,clients_generator=clients_generator)
```

The second option avoids having clients in memory, which may be demanding for cross-device problems.
In caching the data, this optimization is done by the dataloader.
In our thesis, data storage is not to demanding, leading to us caching all data in the dataloader.



### Algorithm.py
The baseclass for all algorithms| defining common methods such as `run` and `get_callback_data` used to iterate through algorithms in the Experiments.
All parameters defined below are used in all algorithms.

| Parameter | Description | Datatype |
| --- | ----------- | --- |
| dataloaders | A list of dataloaders representing all clients used | Dataloader |
| server | The server object used in aggregation| Server |
| clients_sample_alpha | Alpha for dirichlet sampling of clients | float |
| seed | Seed used when sampling clients | int |
|client_generator | Generator function for clients | method |

Further all the following algorithms also have this common hyperparameters:

| Parameter | Description | Datatype |
| --- | ----------- | --- |
| dataloader | A dataloader object used to generate clients| Dataloader |
| Model | A Model module used to intiate all the modes | Model Module |
|batch_size|The batch size used in training| int |
|clients_per_round| The amount of clients sampled in each round | int |
|client_lr| The learning rate on the client side| float |
|momentum| The momentum used on the client side | float |
|decay| The decay used on the client side | float |
|dampening| The dampening used on the client side | float |
|server_optimizer| Which type of optimization done on the server side, either 'none','sgd' or 'adam'| str|
|server_lr| The server learning rate used when running the algorithm.| float |
|tau| The server dampening used| float |
|b1| The b1 coefficent for the adam optimizer| float |
|b2|  The b2 coefficent for the adam optimizer | float |
|server_momentum| The momentum used on the server side | float |

### FedAg.py
Implementation of [FedAg](https://www.mdpi.com/1099-4300/23/1/41) using a `FedAgServer` and `SGDClient`.

### FedAvg.py
Implementation of [FedAvg](https://www.morganclaypool.com/doi/abs/10.2200/S00960ED2V01Y201910AIM043) using a `FedAvgServer` and `SGDClient`.

### FedBe.py
Implementation of [FedBe](https://arxiv.org/abs/2009.01974) using a `FedBeServer` and `SGDClient`.

Parameters:
| Parameter | Description | Datatype |
| --- | ----------- | --- |
| M | Amount of models sampled to the ensamble | int |
swa_lr1 | First SWA learning rate| float|
swa_lr2 | Second SWA learning rate| float|
swa_freq | Frequency of SWA learning rate switch | int |
swa_epochs | Number of SWA epochs made | int |

### FedKp.py
Implementation of FedKp using a `FedKpServer` and `SGDClient`.

Parameters:
| Parameter | Description | Datatype |
| --- | ----------- | --- |
|kernel_function | Choice of kernel function, either 'epanachnikov' or 'gaussian'| str |
|bandwidth_scaling | Amount of bandwidth scaling used | float |
|cluster_mean | Boolean if to cluster clients or not | Boolean |
|bandwidth | choice of bandwidth method, either 'silverman','scotts','plug-in','cross-val','local'| str |
|store_distributions | Boolean i to stor kernel densities | Boolean |
|max_iter| Max iterations of mean-shift |

### FedPa.py
Implementation of [FedPa](https://arxiv.org/abs/2010.05273) using a `FedAvgServer` and `FedPaClient`.

Parameters:
| Parameter | Description | Datatype |
| --- | ----------- | --- |
| burnin | The amount burnin communication rounds | int |
| client_burnin | the amount of epochs of burnin in training | int |
| shrinkage | The shrinkage used for truncated covariance estimation | float |

### FedKpPa.py
Implementation of FedKpPa using a `FedKpServer` and `FedPaClient`.
Uses the same paramters as FedKp and FedPa.

### FedProx.py
Implementation of [FedProx](https://proceedings.mlsys.org/paper/2020/hash/38af86134b65d0f10fe33d30dd76442e-Abstract.html) using a `FedAvgServer` and `FedProxClient`.

Parameters:
| Parameter | Description | Datatype |
| --- | ----------- | --- |
| mu | The penalty parameter used | float |

### SGLD.py
Implementation of [SGLD](https://proceedings.mlr.press/v161/mekkaoui21a.html) using a `SGLDServer` and `SGDClient`.

Parameters:
| Parameter | Description | Datatype |
| --- | ----------- | --- |
| burn_in | Amount of burnin communication rounds used | int |
