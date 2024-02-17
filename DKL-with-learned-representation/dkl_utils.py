# adapted from https://github.com/Wenlin-Chen/ADKF-IFT/blob/main/fs_mol/utils/dkl_utils.py
# adapted from https://github.com/seokhokang/reaction_yield_nn

import torch
import torch.nn as nn
import numpy as np
import time

import gpytorch
import dgl
from scipy import stats

from copy import deepcopy
from torch.optim.lr_scheduler import MultiStepLR
from dgl.nn.pytorch import NNConv, Set2Set

from torch.utils.data import DataLoader
from dgl.data.utils import split_dataset

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from graph_dataset import GraphDataset

def collate_reaction_graphs(batch):

    batchdata = list(map(list, zip(*batch)))
    gs = [dgl.batch(s) for s in batchdata[:-1]]
    labels = torch.FloatTensor(batchdata[-1])
    
    return *gs, labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
 
def run_on_batches(
    model,
    train: bool = False,
):

    n_epochs = 400
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-5)
    lr_scheduler = MultiStepLR(optimizer, milestones = [300, 350], gamma = 0.1, verbose = False)
    clip_value = 0.1

    if train:
        model.train()
    else:
        model.eval()

    task_preds = []
    task_vars = []
    task_labels = []

    train_size = 3164 # [3164, 2767, 1977, 1186, 791, 395, 197, 98]
    batch_size = 128

    train_data = GraphDataset()

    frac_split = (train_size + 1e-5)/len(train_data)
    train_set, test_set = split_dataset(train_data, [frac_split, 1 - frac_split], shuffle=False, random_state=1)
    
    train_loader = DataLoader(dataset=train_set, batch_size=int(np.min([batch_size, len(train_set)])), shuffle=True, collate_fn=collate_reaction_graphs, drop_last=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)

    # training 
    train_y = train_loader.dataset.dataset.yld[train_loader.dataset.indices]
    train_y_mean = np.mean(train_y)
    train_y_std = np.std(train_y)

    try:
        rmol_max_cnt = train_loader.dataset.dataset.rmol_max_cnt
    except:
        rmol_max_cnt = train_loader.dataset.rmol_max_cnt

    # Compute loss at training time
    if train:
        # model.load_state_dict(model.init_params)
        # torch.set_grad_enabled(True)
        start_time = time.time()
        for i in range(n_epochs):
            for batchidx, batchdata in enumerate(train_loader):
                inputs_rmol = [b.to(device) for b in batchdata[:rmol_max_cnt]]
                labels = (batchdata[-1] - train_y_mean) / train_y_std
                labels = labels.to(device)
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
            print('--- training epoch %d, lr %f, loss %.3f, time elapsed(min) %.2f'%(i, optimizer.param_groups[-1]['lr'], train_loss, (time.time()-start_time)/60))
        per_sample_loss = batch_loss.detach()
    
    # compute metric at test time
    else:
        task_labels = []
        task_preds = []
        task_vars = []
    
        nlpd_list = []
        msll_list = []
        qce_list = []

        try:
            rmol_max_cnt_test = test_loader.dataset.dataset.rmol_max_cnt
        except:
            rmol_max_cnt_test = test_loader.dataset.rmol_max_cnt
        
        # Compute task loss
        for batchidx, batchdata in enumerate(test_loader):
            inputs_rmol_test = [b.to(device) for b in batchdata[:rmol_max_cnt_test]]
            labels = (batchdata[-1]).to(device)
            # labels_test = (batchdata[-1]).to(device)
            labels_test = (batchdata[-1] - train_y_mean) / train_y_std
            labels_test = labels_test.to(device)
            batch_logits = model(inputs_rmol_test, labels, train=False)

            # Compute NLPD on the Test set. Raise exception if computation fails
            nlpd = gpytorch.metrics.negative_log_predictive_density(batch_logits, labels_test)
            
            # Compute MSLL on Test set
            msll = gpytorch.metrics.mean_standardized_log_loss(batch_logits, labels_test)

            # Compute quantile coverage error on test set
            qce = gpytorch.metrics.quantile_coverage_error(batch_logits, labels_test, quantile=95)

            with torch.no_grad():
                batch_preds = batch_logits.mean.detach().cpu().numpy()
                batch_vars = batch_logits.variance.detach().cpu().numpy()
                task_labels.append(labels.detach().cpu().numpy())
            
            batch_preds = (batch_preds*train_y_std)+train_y_mean
            task_preds.append(batch_preds)
            task_vars.append(batch_vars)
            
            nlpd_list.append(nlpd)
            msll_list.append(msll)
            qce_list.append(qce)
    
    if train:
        # we will report loss per sample as before.
        metrics = None
    else:
        per_sample_loss = None

        predictions = np.concatenate(task_preds, axis=0)
        variances = np.concatenate(task_vars, axis=0)
        labels=np.concatenate(task_labels, axis=0)
        
        nlpd_list = torch.tensor(nlpd_list)
        msll_list = torch.tensor(msll_list)
        qce_list = torch.tensor(qce_list)

        nlpd = torch.mean(nlpd_list)
        msll = torch.mean(msll_list)
        qce = torch.mean(qce_list)

        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        r2 = r2_score(labels, predictions)
        rmse = np.sqrt(mse)

        print("RMSE: ", rmse)
        print("MAE: ", mae)
        print("R2: ", r2)   
        print("NLPD: ", nlpd)
        print("MSLL: ", msll)
        print("QCE: ", qce)

        metrics=rmse
        
    return per_sample_loss, metrics


def evaluate_dkl_model(model):
    _none1, _none2 = run_on_batches(
        model,
        train=True,
    )

    _, result_metrics = run_on_batches(
        model,
        train=False,
    )
                
    return result_metrics     

    
    
    


        
    