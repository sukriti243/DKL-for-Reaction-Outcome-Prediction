import torch
import torch.nn as nn
import numpy as np
import time

import gpytorch
from gpytorch.distributions import MultivariateNormal

from dataloader import ReactionLoader, transform_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from model import ExactGPLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

class DKL_DRFP_Model(nn.Module):
    def __init__(self, gp_kernel, use_ard, use_lengthscale_prior):
        super().__init__()

        self.gp_kernel = gp_kernel
        self.use_ard = use_ard
        self.use_lengthscale_prior = use_lengthscale_prior

        # Create MLP if needed:
        self.fc_out_dim = 256
        # Determine dimension:
        fc_in_dim = 2048

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
        self.gp_model = ExactGPLayer(
            train_x=dummy_train_x, train_y=dummy_train_y, likelihood=self.gp_likelihood, 
            kernel=kernel_type, ard_num_dims=ard_num_dims
        )

        if use_lengthscale_prior:
            scale = 0.25
            loc = 0.0
            lengthscale_prior = gpytorch.priors.LogNormalPrior(loc=loc, scale=scale)
            self.gp_model.covar_module.base_kernel.register_prior(
                "lengthscale_prior", lengthscale_prior, lambda m: m.lengthscale, lambda m, v: m._set_lengthscale(v)
            )
            self.gp_model.covar_module.base_kernel.lengthscale = torch.ones_like(self.gp_model.covar_module.base_kernel.lengthscale) * lengthscale_prior.mean

        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_likelihood, self.gp_model)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def save_gp_params(self):
        self.gp_model_params = deepcopy(self.gp_model.state_dict())
        self.gp_likelihood_params = deepcopy(self.gp_likelihood.state_dict())

    def load_gp_params(self):
        self.gp_model.load_state_dict(self.gp_model_params)
        self.gp_likelihood.load_state_dict(self.gp_likelihood_params)

    def forward(self, batch, train: bool):
        
        self.fc = self.fc.double()

        support_features, support_target, query_features, query_target = batch
        support_features_flat = self.fc(support_features)
        query_features_flat = self.fc(query_features)

        if self.training and train:
            self.gp_model.set_train_data(inputs=support_features_flat, targets=support_target, strict=False)
            logits = self.gp_model(support_features_flat)
            
        else:
            assert self.training == False and train == False
            self.gp_model.train()

            self.gp_model.set_train_data(inputs=support_features_flat.detach(), targets=support_target, strict=False)
            
            self.gp_model.eval()
            self.gp_likelihood.eval()
            
            with torch.no_grad():
                logits = self.gp_likelihood(self.gp_model(query_features_flat))
            
        return logits

    def compute_loss(self, logits: MultivariateNormal) -> torch.Tensor:
        assert self.training == True
        return -self.mll(logits, self.gp_model.train_targets)
 
def run_on_batches():

    # initialise performance metric lists
    r2_list = []
    rmse_list = []
    mae_list = []
    nlpd_list = []
    msll_list = []

    n_trials = 10

    # Load the dataset

    loader = ReactionLoader()
    loader.load_benchmark("DreherDoyleRXN", "/homes/ss2971/Documents/GNN-GP/DKL/dreher_doyle_science_aar5169.csv")

    # Featurise the molecules. 
    loader.featurize('drfp', nBits=2048)
    X = loader.features
    y = loader.labels

    print('\nBeginning training loop...')

    for i in range(0, n_trials):
        
        print(f'Starting trial {i}')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        model = DKL_DRFP_Model(gp_kernel='matern', use_ard=False, use_lengthscale_prior=False).to(device)

        # Set parameters
        n_epochs = 400
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-5)
        lr_scheduler = None #MultiStepLR(optimizer, milestones = [300, 350], gamma = 0.1, verbose = False)
        clip_value = None
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

        #  We standardise the outputs but leave the inputs unchanged
        y_train = np.array(y_train)
        y_train = y_train.reshape(-1,1)
        y_test = np.array(y_test)
        y_test = y_test.reshape(-1,1)
        
        _, y_train, _, y_test, y_scaler = transform_data(X_train, y_train, X_test, y_test)

        X_train = torch.tensor(X_train).to(device)
        X_test = torch.tensor(X_test).to(device)
        y_train = torch.tensor(y_train).flatten().to(device)
        y_test = torch.tensor(y_test).flatten().to(device)

        batch = (X_train, y_train, X_test, y_test)

        start_time = time.time()
        for i in range(n_epochs):
            optimizer.zero_grad()
            # Compute task loss
            model.train()
            batch_logits = model(batch, train=True)
            batch_loss = model.compute_loss(batch_logits)
            batch_loss.backward()
            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()

            train_loss = batch_loss.detach().item()
            if lr_scheduler is not None:
                lr_scheduler.step()

            print('--- training epoch %d, lr %f, loss %.3f, time elapsed(min) %.2f'%(i, optimizer.param_groups[-1]['lr'], train_loss, (time.time()-start_time)/60))

        # Compute metric at test time
        # Compute task loss
        model.eval()
        batch_logits = model(batch, train=False)

        # Compute NLPD on the Test set. Raise exception if computation fails
        try:
            nlpd = gpytorch.metrics.negative_log_predictive_density(batch_logits, y_test)
        except:
            Exception(f'NLPD calculation failed on trial {i}')
            continue

        # Compute MSLL on Test set
        msll = gpytorch.metrics.mean_standardized_log_loss(batch_logits, y_test)

        with torch.no_grad():
            batch_preds = batch_logits.mean
            batch_preds = y_scaler.inverse_transform(batch_preds.detach().cpu().unsqueeze(dim=1))
            y_test = y_scaler.inverse_transform(y_test.detach().cpu().unsqueeze(dim=1))

        # Compute R^2, RMSE and MAE on Test set
        score = r2_score(y_test, batch_preds)
        rmse = np.sqrt(mean_squared_error(y_test, batch_preds))
        mae = mean_absolute_error(y_test, batch_preds)

        print('rmse: ', rmse)

        r2_list.append(score)
        rmse_list.append(rmse)
        mae_list.append(mae)

        nlpd_list.append(nlpd)
        msll_list.append(msll)

    r2_list = np.array(r2_list)
    rmse_list = np.array(rmse_list)
    mae_list = np.array(mae_list)

    nlpd_list = torch.tensor(nlpd_list)
    msll_list = torch.tensor(msll_list)
    
    nlpd = torch.mean(nlpd_list)
    msll = torch.mean(msll_list)
    
    print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list)/np.sqrt(len(r2_list))))
    print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list), np.std(rmse_list)/np.sqrt(len(rmse_list))))
    print("mean MAE: {:.4f} +- {:.4f}\n".format(np.mean(mae_list), np.std(mae_list)/np.sqrt(len(mae_list)))) 
        
    return None


# For running the experiment
run_on_batches()
    
    


        
    
