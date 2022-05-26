# Models
A selection of the different models used in the thesis.
All models except StackOverflow_Model is a `nn_model` being a wrapper around pytorch `torch.nn.Module` with added methods used in the experiements.
All models inheriting nn_model just define the architecture used for the model, with the other methods are inherited.
The model architecture is specified in the Appendix and can be found in each subsequent file.
