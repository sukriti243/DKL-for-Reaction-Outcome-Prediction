# adapted from https://github.com/Wenlin-Chen/ADKF-IFT/blob/main/fs_mol/models/dkl.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import gpytorch
from gpytorch.distributions import MultivariateNormal

from graph_dataset import GraphDataset
from model import GraphFeatureExtractor, ExactGPLayer

data = GraphDataset()
node_dim = data.rmol_node_attr[0].shape[1]
edge_dim = data.rmol_edge_attr[0].shape[1]

class DKLModel(nn.Module):
    
    def __init__(self, gp_kernel, use_ard, use_lengthscale_prior):
        
        super().__init__()

        self.gp_kernel = gp_kernel
        self.use_ard = use_ard
        self.use_lengthscale_prior = use_lengthscale_prior
        self.fc_out_dim = 128
        
        # Create GNN if needed
        self.graph_feature_extractor = GraphFeatureExtractor(node_dim, edge_dim)

        # Create MLP:
        fc_in_dim = 512

        self.fc = nn.Sequential(
            nn.Linear(fc_in_dim, 256), 
            nn.PReLU(), nn.Dropout(0.1),
            nn.Linear(256, self.fc_out_dim)
        )

        if self.use_ard:
            ard_num_dims = self.fc_out_dim
        else:
            ard_num_dims = None
        self.__create_tail_GP(kernel_type=self.gp_kernel, ard_num_dims=ard_num_dims, use_lengthscale_prior=self.use_lengthscale_prior)

    def __create_tail_GP(self, kernel_type, ard_num_dims=None, use_lengthscale_prior=False):
        dummy_train_x = torch.ones(64, self.fc_out_dim)
        dummy_train_y = torch.ones(64)

        self.gp_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.gp_model = ExactGPLayer(train_x=dummy_train_x, train_y=dummy_train_y, likelihood=self.gp_likelihood, kernel=kernel_type, ard_num_dims=ard_num_dims)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_likelihood, self.gp_model)
        
        if use_lengthscale_prior:
            scale = 0.25
            loc = 0.0
            lengthscale_prior = gpytorch.priors.LogNormalPrior(loc=loc, scale=scale)
            self.gp_model.covar_module.base_kernel.register_prior(
                "lengthscale_prior", lengthscale_prior, lambda m: m.lengthscale, lambda m, v: m._set_lengthscale(v)
            )
            self.gp_model.covar_module.base_kernel.lengthscale = torch.ones_like(self.gp_model.covar_module.base_kernel.lengthscale) * lengthscale_prior.mean  

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def forward(self, rmol, target, train: bool):
        
        features = self.graph_feature_extractor(rmol)
        features_flat = self.fc(features)
      
        if self.training and train:
            self.gp_model.set_train_data(inputs=features_flat, targets=target, strict=False)
            logits = self.gp_model(features_flat)
            
        else:
            assert self.training == False and train == False
            
            self.gp_model.eval()
            self.gp_likelihood.eval()
            
            with torch.no_grad():
                logits = self.gp_likelihood(self.gp_model(features_flat))

        return logits
    
    def compute_loss(self, logits: MultivariateNormal) -> torch.Tensor:
        assert self.training == True
        return -self.mll(logits, self.gp_model.train_targets)
