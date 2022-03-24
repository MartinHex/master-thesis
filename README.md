# Master Thesis
Repo for the master thesis done by Oscar Johansson and Martin Gullbrandson

# Setup

Before running the repository make sure to install all required dependencies by creating a conda environment.
Running the following will create an environment called 'thesis'.
```
conda env create -f environment.yml
```
activate the thesis environment by
```
conda activate thesis
```
and you should now be able to run all scripts associated with our thesis project.
To deactivate the environment execute the following.
```
conda deactivate
```

To run an example with FedAvg simply execute:

```
python FedAvgExample.py
```
Doing will train a simple network on the MNIST data split into 5 IID clients.

## Using Jupyter Notebooks

Some analysis is done in notebooks which produce a lot of metadata. If you intend to run, change and commit anything from notebooks please also install the `nbstripout` package by running

```
nbstripout --install
```
