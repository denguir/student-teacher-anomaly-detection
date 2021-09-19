import torch
import torch.nn as nn


def load_model(model, model_path):
    model_name = model_path.split('/')[-1]
    try:
        print(f'Loading of {model_name} succesful.')
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError as e:
        print(e)
        print('No model available.')
        print(f'Initilialisation of random weights for {model_name}.')


def increment_mean_and_var(mu_N, var_N, N, batch):
    '''Increment value of mean and variance based on
       current mean, var and new batch
    '''
    # batch: (batch, h, w, vector)
    B = batch.size()[0] # batch size
    # we want a descriptor vector -> mean over batch and pixels
    mu_B = torch.mean(batch, dim=[0,1,2])
    S_B = B * torch.var(batch, dim=[0,1,2], unbiased=False) 
    S_N = N * var_N
    mu_NB = N/(N + B) * mu_N + B/(N + B) * mu_B
    S_NB = S_N + S_B + B * mu_B**2 + N * mu_N**2 - (N + B) * mu_NB**2
    var_NB = S_NB / (N+B)
    return mu_NB, var_NB, N + B


def mc_dropout(model):
	for module in model.modules():
		if isinstance(module, nn.Dropout)\
            or isinstance(module, nn.Dropout2d)\
            or isinstance(module, nn.Dropout3d):
			module.train()
