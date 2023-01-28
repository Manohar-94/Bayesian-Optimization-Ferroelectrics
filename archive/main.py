import sys
import numpy as np
import torch
import gpytorch
import wandb
from model import GridGP
from archive.datasetmaker import dataset
from acq_funcs import EI, PI, cust_acq, thompson

###### SWEEPS ########
config_defaults = {
    "epochs": 1000,
    "kernel": "rbf",
    "lr": 0.1,
    "lscale_1": 5,
    "lscale_2": 10,
    "lscale_3": 0.5,
    "lscale_4": 1,
    "noise": 3.0
}
wandb.init(config=config_defaults)
config = wandb.config

def kernel_func(config_kernel, num_params):
    if config_kernel == "rbf":
        return gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=num_params))

def make_model(train_x, train_y, num_params, config):
    kernel = kernel_func(config.kernel, num_params)
    noises = config.noise * torch.ones(len(train_x))
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noises)
    model = GridGP(train_x, train_y, likelihood, kernel)
    lscale = [config.lscale_1, config.lscale_2, config.lscale_3, config.lscale_4]
    model.covar_module.base_kernel.lengthscale = torch.tensor(lscale)
    return likelihood, model

def train(train_x, train_y, num_params, config):
    likelihood, model = make_model(train_x, train_y, num_params, config)
    training_iter = config.epochs

    # Place both the model and likelihood in training mode
    model.train(), likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)

        # backpropogate error
        wandb.define_metric("train loss", summary="min")
        loss = -mll(output, train_y)
        wandb.log({"train loss": loss.item()})
        loss.backward()

        if i % 100 == 0: 
            print('Iter %d/%d - Loss: %.3f  lengthscale1: %s   noise: %s' % (
                    i+1, training_iter, loss.item(), 
                    model.covar_module.base_kernel.lengthscale.detach().numpy(),
                    model.likelihood.noise.detach().numpy()
                    )) 
        optimizer.step()
    return likelihood, model

def eval(likelihood, model, test_x):
    # make predictions (whether by long or short form)
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # also use mll from short form 
        obs = likelihood(model(test_x), noise=(torch.ones(len(test_x))*5))
    return obs

def acq(obs, train_y, test_grid, n=30):
    # Evaluate acquisition functions on current predictions (observations)
    bounds = [1,1,1,1]
    nshape = (n,n,n,n)
    
    # Probability of Improvement
    PI_acq = PI(obs, bounds, train_y)
    PI_acq_shape = PI_acq.detach().numpy().reshape(nshape).T
    
    # Expected Improvement
    EI_acq = EI(obs, bounds, train_y)
    EI_acq_shape = EI_acq.detach().numpy().reshape(nshape).T

    ei = np.unravel_index(EI_acq_shape.argmax(), EI_acq_shape.shape)
    pi = np.unravel_index(PI_acq_shape.argmax(), PI_acq_shape.shape)

    pred_var = obs.variance.view(nshape).detach().numpy().T
    lower, upper = obs.confidence_region()
    upper_surf = upper.detach().numpy().reshape(nshape).T

    ucb = np.unravel_index(upper_surf.argmax(), upper_surf.shape)
    max_var = np.unravel_index(pred_var.argmax(), pred_var.shape)

    print(test_grid[ei[1], 0], test_grid[ei[0], 1])
    print(test_grid[pi[1], 0], test_grid[pi[0], 1])
    print(test_grid[ucb[1], 0], test_grid[ucb[0], 1])
    print(test_grid[max_var[1], 0], test_grid[max_var[0], 1])

def main():
    train_x, train_y, num_params, test_grid, test_x = dataset()
    likelihood, model = train(train_x, train_y, num_params, config)
    obs = eval(likelihood, model, test_x)
    acq(obs, train_y, test_grid)
    
if __name__ == "__main__":
    main()
