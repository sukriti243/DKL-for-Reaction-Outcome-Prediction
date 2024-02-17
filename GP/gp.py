# adapted from https://github.com/leojklarner/gauche/blob/main/notebooks/GP%20Regression%20on%20Molecules.ipynb
import torch
import gpytorch
import numpy as np
import pandas as pd
from botorch import fit_gpytorch_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from dataloader import ReactionLoader, transform_data

import warnings
warnings.filterwarnings("ignore") # Turn off GPyTorch warnings

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
# Regression experiments parameters, number of random splits and split size

n_trials = 10
test_set_size = 0.3

# Load the dataset
loader = ReactionLoader()
loader.load_benchmark("DreherDoyleRXN", "/homes/ss2971/Documents/GNN-GP/DKL/dreher_doyle_science_aar5169.csv")

# Featurise the molecules. 
loader.featurize('drfp', nBits=1024)
X = loader.features
y = loader.labels

def evaluate_model(X, y):
    """
    Helper function for model evaluation
    
    X: Inputs
    y: Outputs
    """

    # initialise performance metric lists
    r2_list = []
    rmse_list = []
    mae_list = []
    nlpd_list = []
    msll_list = []
    qce_list = []
    
    print('\nBeginning training loop...')

    for i in range(0, n_trials):
        
        print(f'Starting trial {i}')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size, random_state=i)

        X_train = np.array(X_train)
        X_test = np.array(X_test)

        #  We standardise the outputs but leave the inputs unchanged
        y_train = np.array(y_train)
        y_train = y_train.reshape(-1,1)
        y_test = np.array(y_test)
        y_test = y_test.reshape(-1,1)
        
        _, y_train, _, y_test, y_scaler = transform_data(X_train, y_train, X_test, y_test)

        # Convert numpy arrays to PyTorch tensors and flatten the label vectors
        X_train = torch.tensor(X_train.astype(np.float64))
        X_test = torch.tensor(X_test.astype(np.float64))
        y_train = torch.tensor(y_train).flatten()
        y_test = torch.tensor(y_test).flatten()

        # initialise GP likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(X_train, y_train, likelihood)

        # Find optimal model hyperparameters
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # Use the BoTorch utility for fitting GPs in order to use the LBFGS-B optimiser (recommended)
        fit_gpytorch_model(mll)

        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()

        # full GP predictive distribution
        trained_pred_dist = likelihood(model(X_test))

        # mean and variance GP prediction
        f_pred = model(X_test)

        y_pred = f_pred.mean
        y_var = f_pred.variance

         # Compute NLPD on the Test set. Raise exception if computation fails
        try:
            nlpd = gpytorch.metrics.negative_log_predictive_density(trained_pred_dist, y_test)
        except:
            Exception(f'NLPD calculation failed on trial {i}')
            continue

        # Compute MSLL on Test set
        msll = gpytorch.metrics.mean_standardized_log_loss(trained_pred_dist, y_test)

        # Compute quantile coverage error on test set
        qce = gpytorch.metrics.quantile_coverage_error(trained_pred_dist, y_test, quantile=95)

        # Transform back to real data space to compute metrics and detach gradients. Must unsqueeze dimension
        # to make compatible with inverse_transform in scikit-learn version > 1
        y_pred = y_scaler.inverse_transform(y_pred.detach().unsqueeze(dim=1))
        y_test = y_scaler.inverse_transform(y_test.detach().unsqueeze(dim=1))

        # Compute R^2, RMSE and MAE on Test set
        score = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        r2_list.append(score)
        rmse_list.append(rmse)
        mae_list.append(mae)
        
        nlpd_list.append(nlpd)
        msll_list.append(msll)
        qce_list.append(qce)
        
    r2_list = np.array(r2_list)
    rmse_list = np.array(rmse_list)
    mae_list = np.array(mae_list)
    
    nlpd_list = torch.tensor(nlpd_list)
    msll_list = torch.tensor(msll_list)
    qce_list = torch.tensor(qce_list)
   
    print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list)/np.sqrt(len(r2_list))))
    print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)/np.sqrt(len(rmse_list))))
    print("mean MAE: {:.4f} +- {:.4f}\n".format(np.mean(mae_list), np.std(mae_list)/np.sqrt(len(mae_list)))) 

    
# To run the experiments 
evaluate_model(X, y)
