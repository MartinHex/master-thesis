# clients
The different types clients studied in this thesis.
The design is taken from [FedOpt](https://www.mdpi.com/2076-3417/10/8/2864) seeing Federated learning as 2 optimization problems, client optimization and a server optimization.

### Base_Client.py
The base class for all clients, defining basic functionalities such as ´get_weights´, `set_weights` etc.
All parameters defined below are used by the inheriting classes.

Parameters:
| Parameter | Description | Datatype |
| --- | ----------- | --- |
| Model | A Model used by the client | Model |

### SGDClient.py
The SGDClient uses [SGD](https://link.springer.com/chapter/10.1007/978-3-7908-2604-3_16) optimization on the client-side for local training.
This is used as a further base class for many of the other algorithms.
All parameters defined below are used by the inheriting classes.

Parameters:
| Parameter | Description | Datatype |
| --- | ----------- | --- |
| dataloader | A dataloader used by the client | Dataloader |
|learning_rate| The learning rate on the client side| float |
|momentum| The momentum used on the client side | float |
|decay| The decay used on the client side | float |
|dampening| The dampening used on the client side | float |

### FedPaClient .py
Clients using the [FedPa](https://arxiv.org/abs/2010.05273) optimization in training the algorithm.
Implementation is inspired by their example [repo](https://github.com/alshedivat/fedpa/blob/master/federated/inference/local.py).

Parameters:
| Parameter | Description | Datatype |
| --- | ----------- | --- |
| burn_in | the amount of epochs of burnin in training | int |
| shrinkage | The shrinkage used for truncated covariance estimation | float |

### FedProxClient.py
[FedProx](https://proceedings.mlsys.org/paper/2020/hash/38af86134b65d0f10fe33d30dd76442e-Abstract.html) adds a penalty L2 regularization term to the local optimization done.
This stops the local client from deviating to much from the original broadcasted weight.

Parameters:
| Parameter | Description | Datatype |
| --- | ----------- | --- |
| mu | The penalty parameter used | float |
