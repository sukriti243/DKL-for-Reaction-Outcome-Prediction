# adapted from https://github.com/leojklarner/gauche/blob/main/notebooks/Bayesian%20Optimisation%20Over%20Molecules.ipynb
import time
import warnings
warnings.filterwarnings("ignore") # Turn off Graphein warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from botorch.exceptions import BadInitialCandidatesWarning

from scipy import stats
from scipy.stats import norm

import gpytorch
from gpytorch.distributions import MultivariateNormal
from sklearn.model_selection import train_test_split

from model import ExactGPLayer
from dataloader import ReactionLoader, transform_data

class DKLModel(nn.Module):
    
    def __init__(self, gp_kernel, num_outputs, use_ard, use_lengthscale_prior):
        
        super().__init__()

        self.gp_kernel = gp_kernel
        self.num_outputs = num_outputs 
        self.use_ard = use_ard
        self.use_lengthscale_prior = use_lengthscale_prior
        self.fc_out_dim = 256
        
        # Create MLP:
        fc_in_dim = 1024

        self.fc = nn.Sequential(
            nn.Linear(fc_in_dim, 512), 
            nn.PReLU(), nn.Dropout(0.1),
            nn.Linear(512, self.fc_out_dim)
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
    
    def forward(self, features, target, train: bool):
        self.fc = self.fc.double()
        
        features_flat = self.fc(features)
        
        if self.training and train:
            self.gp_model.set_train_data(inputs=features_flat, targets=target, strict=False)
            logits = self.gp_model(features_flat)
            # logits = -self.mll(logits, target)
            
        else:
            assert train == False
        
            self.gp_model.eval()
            self.gp_likelihood.eval()
            with torch.no_grad():
                logits = self.gp_likelihood(self.gp_model(features_flat))

        return logits
    
    def compute_loss(self, logits: MultivariateNormal) -> torch.Tensor:
        assert self.training == True
        return -self.mll(logits, self.gp_model.train_targets)
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_model(state_dict=None):
    """
    Initialise model and loss function.

    Args:
        state_dict: current state dict used to speed up fitting

    Returns: model object
    """

    # define model for objective
    model = DKLModel(gp_kernel='matern', num_outputs=1, use_ard=False, use_lengthscale_prior=False)
    
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model

def run_on_batches(
    model,
    inputs_rmol,
    labels,
    n_epochs,
    train: bool = True,
):

    n_epochs = 300
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay=1e-5)
    lr_scheduler = None #torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [300, 350], gamma = 0.1, verbose = False)
    clip_value = None #0.1

    if train:
        model.train()
    else:
        model.eval()

    # Compute loss at training time
    if train:
        start_time = time.time()
        for i in range(n_epochs):

            optimizer.zero_grad()
            # Compute task loss
            batch_logits = model(inputs_rmol, labels, train=True)
            batch_loss = model.compute_loss(batch_logits)
            batch_loss.backward()
            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()

            train_loss = batch_loss.detach().item()
            if lr_scheduler is not None:
                lr_scheduler.step()
            # print('--- training epoch %d, lr %f, loss %.3f, time elapsed(min) %.2f'%(i, optimizer.param_groups[-1]['lr'], train_loss, (time.time()-start_time)/60))
        per_sample_loss = batch_loss.detach()

    return model
    
def initialize_model(state_dict=None):
    """
    Initialise model and loss function.

    Args:
        state_dict: current state dict used to speed up fitting

    Returns: model object
    """

    # define model for objective
    model = DKLModel(gp_kernel='matern', num_outputs=1, use_ard=False, use_lengthscale_prior=False)
    
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model

def expected_improvement(model, heldout_inputs, y_best):
   
    output = model(heldout_inputs, None, train=False)
    mu = output.mean.detach().cpu().numpy()
    std = output.stddev.detach().cpu().numpy()
    y_best = y_best.detach().cpu().numpy()

    with np.errstate(divide='warn'):
        imp = mu - y_best
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
    
    return ei

def optimize_acqf_and_get_observation(model, y_best, heldout_inputs, heldout_outputs):
    """
    Optimizes the acquisition function, and returns a new candidate and an observation.

    Args:
        acq_func: Object representing the acquisition function
        heldout_points: Tensor of heldout points

    Returns: new_x, new_obj
    """

    # Loop over the discrete set of points to evaluate the acquisition function at.

    acq_vals = []

    for i in range(len(heldout_outputs)):
        acq_vals.append(expected_improvement(model, heldout_inputs[i].unsqueeze(-2), y_best))
       
    # observe new values
    acq_vals = torch.tensor(acq_vals)
    best_idx = torch.argmax(acq_vals)
    new_x = heldout_inputs[best_idx].unsqueeze(-2)  # add batch dimension
    new_obj = heldout_outputs[best_idx].unsqueeze(-1)  # add output dimension

    # Delete the selected input and value from the heldout set.
    heldout_inputs = torch.cat((heldout_inputs[:best_idx], heldout_inputs[best_idx+1:]), axis=0)
    heldout_outputs = torch.cat((heldout_outputs[:best_idx], heldout_outputs[best_idx+1:]), axis=0)

    return new_x, new_obj, heldout_inputs, heldout_outputs

# Bayesian optimisation experiment parameters, number of random trials, split size, batch size and number of iterations of Bayesian optimisation

N_TRIALS = 10
N_ITERS = 20
verbose = False

# Load the  dataset
loader = ReactionLoader()
loader.load_benchmark("DreherDoyleRXN", "/homes/ss2971/Documents/GNN-GP/DKL/dreher_doyle_science_aar5169.csv")

# We use the fragprints representations (a concatenation of Morgan fingerprints and RDKit fragment features)
loader.featurize('drfp')
X = loader.features
y = loader.labels

warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

best_observed_all_ei, best_random_all = [], []

# average over multiple random trials (each trial splits the initial training set for the GP in a random manner)
for trial in range(1, N_TRIALS + 1):

    print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
    best_observed_ei, best_random = [], []

    # Generate initial training data and initialize model
    train_x_ei, heldout_x_ei, train_y_ei, heldout_y_ei = train_test_split(X, y, test_size=0.95, random_state=trial)

    #  We standardise the outputs but leave the inputs unchanged
    train_y_ei = np.array(train_y_ei)
    heldout_y_ei = np.array(heldout_y_ei)
    best_observed_value_ei = torch.tensor(np.max(train_y_ei))

    train_x_ei = torch.tensor(train_x_ei.astype(np.float64)).to(device)
    heldout_x_ei = torch.tensor(heldout_x_ei.astype(np.float64)).to(device)
    train_y_ei = torch.tensor(train_y_ei).flatten().to(device)
    heldout_y_ei = torch.tensor(heldout_y_ei).flatten().to(device)
    
    # The initial heldout set is the same for random search
    heldout_x_random = heldout_x_ei
    heldout_y_random = heldout_y_ei

    model = initialize_model()
    model = model.to(device) 

    best_observed_ei.append(best_observed_value_ei)

    # run N_ITERS rounds of BayesOpt after the initial random batch
    for iteration in range(1, N_ITERS + 1):

        t0 = time.time()

        # fit the model
        n_epochs=300
        model_ei = run_on_batches(model, train_x_ei, train_y_ei, n_epochs, train=True)

        # Use analytic acquisition function for batch size of 1
        y_best = torch.max(train_y_ei)
        new_x_ei, new_obj_ei, heldout_x_ei, heldout_y_ei = optimize_acqf_and_get_observation(model_ei, y_best, heldout_x_ei, heldout_y_ei)
        
        # update training points
        train_x_ei = torch.cat([train_x_ei, new_x_ei])
        train_y_ei = torch.cat([train_y_ei, new_obj_ei])

        best_value_ei = max(best_observed_ei[-1], new_obj_ei)
        b = best_value_ei.squeeze(0)
        b1 = b.squeeze(0)
        best_observed_ei.append(b1)

        # reinitialise the model so it is ready for fitting on the next iteration
        # use the current state dict to speed up fitting
        model_ei = initialize_model()
        # model_ei = initialize_model(state_dict = model_ei.state_dict())
        model_ei = model_ei.to(device) 

        t1 = time.time()
        print(f"time = {(t1 - t0)/60:>4.2f}.")
        if verbose:
            print(
                f"\nBatch {iteration:>2}: best_value (random, qEI) = "
                f"({max(best_random):>4.2f}, {best_value_ei:>4.2f}), "
                f"time = {t1 - t0:>4.2f}.", end=""
            )
            
            print(
                f"\nBatch {iteration:>2}: best_value (qEI) = "
                f"({best_value_ei:>4.2f}), "
                f"time = {t1 - t0:>4.2f}.", end=""
            )
        else:
            print("-----", end="")
        print('TRIAL', trial, 'ITERATION:', iteration, 'best_value:', best_value_ei)

    best_observed_all_ei.append(best_observed_ei)



N_TRIALS = 50
y_ei = torch.tensor(best_observed_all_ei).detach().cpu().numpy()
y_ei_mean = y_ei.mean(axis=0)
y_ei_std = y_ei.std(axis=0) / np.sqrt(N_TRIALS)

lower_ei = y_ei_mean - y_ei_std
upper_ei = y_ei_mean + y_ei_std

iters = np.arange(N_ITERS + 1)
plt.plot(iters, y_ei_mean, marker='o', label='FP_DKL')
plt.fill_between(iters, lower_ei, upper_ei, alpha=0.2)

plt.xlabel('Number of Iterations')
plt.ylabel('Best Objective Value')
plt.legend(loc="lower right")
plt.xticks(list(np.arange(1, 21)))
plt.savefig('fp_dkl.png')
plt.show()