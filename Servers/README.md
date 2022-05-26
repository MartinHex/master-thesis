# Servers
The different types servers studied in this thesis.
The design is taken from [FedOpt](https://www.mdpi.com/2076-3417/10/8/2864) seeing Federated learning as 2 optimization problems, client optimization and a server optimization.

# ABCServer.py
The base class for all servers defining the
Here all server optimization using `SGD` or `Adam` found in the optimizers is found.
The class is abstract and can only be inherited requiring a combine function being how the Server algorithm combine client models.
All servers basically define how such aggregation should be done and ABCServer handles all other useful methods.
All parameters defined below are used in all servers which inherit ABCServer.

Parameters:
| Parameter | Description | Datatype |
| --- | ----------- | --- |
| Model | A Model used by the client | Model |
|optimizer| Which type of optimization done on the server side, either 'none','sgd' or 'adam'| str|
|lr| The server learning rate used when running the algorithm.| float |
|tau| The server dampening used| float |
|b1| The b1 coefficent for the adam optimizer| float |
|b2|  The b2 coefficent for the adam optimizer | float |
|momentum| The momentum used on the server side | float |

# ProbabilisticServer.py
A further abstract class around ABCServer being probabilistic servers having a `sample` method from the distribution.
This allows models to be sampled from the distribtion, which would be an interesting approach.
This was an idea in the intial part of the thesis, but was not elaborated on later.

# FedAgServer.py
Implementation of [FedAg](https://www.mdpi.com/1099-4300/23/1/41) using gaussian noise around the average to inference a new weight.

# FedAvgServer.py
Implementation of [FedAvg](https://www.morganclaypool.com/doi/abs/10.2200/S00960ED2V01Y201910AIM043)  taking average over all client models provided.

# FedBeServer.py
Implementation of [FedBe](https://arxiv.org/abs/2009.01974) using a bayesian ensamble and local  data to distilate the client models.

Parameters:
| Parameter | Description | Datatype |
| --- | ----------- | --- |
| M | Amount of models sampled to the ensamble | int |
swa_lr1 | First SWA learning rate| float|
swa_lr2 | Second SWA learning rate| float|
swa_freq | Frequency of SWA learning rate switch | int |
swa_epochs | Number of SWA epochs made | int |


# FedKpServer
Our implementation of FedKp with a series of hyperparamters such as 'cluster_mean', 'bandwidth_scaling' 'kernel_function'.
Also has methods for doing kernel esstimation through `store_distributions` which by default is False to to all storage needed in esstimation.
This method also defines the mean-shift algorithm which is used for inference.

Parameters:
| Parameter | Description | Datatype |
| --- | ----------- | --- |
|kernel_function | Choice of kernel function, either 'epanachnikov' or 'gaussian'| str |
|bandwidth_scaling | Amount of bandwidth scaling used | float |
|cluster_mean | Boolean if to cluster clients or not | Boolean |
|bandwidth | choice of bandwidth method, either 'silverman','scotts','plug-in','cross-val','local'| str |
|store_distributions | Boolean i to stor kernel densities | Boolean |
|max_iter| Max iterations of mean-shift |

# SGLDServer

Parameters:
| Parameter | Description | Datatype |
| --- | ----------- | --- |
| burn_in | Amount of burnin communication rounds used | int |
