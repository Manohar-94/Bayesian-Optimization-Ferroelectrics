import gpytorch
import torch
from botorch.models.gpytorch import GPyTorchModel

class GridGP(gpytorch.models.ExactGP, GPyTorchModel):
    _num_outputs = 1
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(GridGP, self).__init__(train_x, train_y, likelihood)  
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
