# adapted from https://github.com/seokhokang/reaction_yield_nn
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

import dgl
from dgl.nn.pytorch import NNConv, Set2Set

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class MPNN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats, hidden_feats = 64,
                 num_step_message_passing = 3, num_step_set2set = 3, num_layer_set2set = 1,
                 readout_feats = 512):
        
        super(MPNN, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, hidden_feats), nn.ReLU()
        )
        
        self.num_step_message_passing = num_step_message_passing
        
        edge_network = nn.Linear(edge_in_feats, hidden_feats * hidden_feats)
        
        self.gnn_layer = NNConv(
            in_feats = hidden_feats,
            out_feats = hidden_feats,
            edge_func = edge_network,
            aggregator_type = 'sum'
        )
        
        self.activation = nn.ReLU()
        
        self.gru = nn.GRU(hidden_feats, hidden_feats)

        self.readout = Set2Set(input_dim = hidden_feats * 2,
                               n_iters = num_step_set2set,
                               n_layers = num_layer_set2set)

        self.sparsify = nn.Sequential(
            nn.Linear(hidden_feats * 4, readout_feats), nn.PReLU()
        )
             
    def forward(self, g):
            
        node_feats = g.ndata['attr']
        edge_feats = g.edata['edge_attr']
        
        node_feats = self.project_node_feats(node_feats)
        hidden_feats = node_feats.unsqueeze(0)

        node_aggr = [node_feats]        
        for _ in range(self.num_step_message_passing):
            node_feats = self.activation(self.gnn_layer(g, node_feats, edge_feats)).unsqueeze(0)
            node_feats, hidden_feats = self.gru(node_feats, hidden_feats)
            node_feats = node_feats.squeeze(0)
        
        node_aggr.append(node_feats)
        node_aggr = torch.cat(node_aggr, 1)
        
        readout = self.readout(g, node_aggr)
        graph_feats = self.sparsify(readout)
        
        return graph_feats


class reactionMPNN(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats,
                 readout_feats = 512,
                 predict_hidden_feats = 256, prob_dropout = 0.1):
        super(reactionMPNN, self).__init__()
        self.mpnn = MPNN(node_in_feats, edge_in_feats)
        self.predict = nn.Sequential(
            nn.Linear(readout_feats, predict_hidden_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats, 1)
        )
    
    def forward(self, rmols):
        r_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in rmols]), 0)
        out = self.predict(r_graph_feats)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")           
def training(net, train_loader, train_y_mean, train_y_std, cuda = torch.device('cuda:0')):

    train_size = train_loader.dataset.__len__()
    batch_size = train_loader.batch_size
    
    # training 
    train_y = train_loader.dataset.dataset.yld[train_loader.dataset.indices]
    train_y_mean = np.mean(train_y)
    train_y_std = np.std(train_y)
    
    try:
        rmol_max_cnt = train_loader.dataset.dataset.rmol_max_cnt
    except:
        rmol_max_cnt = train_loader.dataset.rmol_max_cnt

    loss_fn = nn.MSELoss()

    n_epochs = 400
    optimizer = Adam(net.parameters(), lr = 1e-3, weight_decay = 1e-5)
    lr_scheduler = MultiStepLR(optimizer, milestones = [300, 350], gamma = 0.1, verbose = False)

    start_time = time.time()
    for epoch in range(n_epochs):
        # training
        net.train()
        for batchidx, batchdata in enumerate(train_loader):
            inputs_rmol = [b.to(device) for b in batchdata[:rmol_max_cnt]]
            
            labels = (batchdata[-1] - train_y_mean) / train_y_std
            labels = labels.to(device)
            
            net = net.to(device)
            pred = net(inputs_rmol)
        
            loss = loss_fn(pred.squeeze(), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss = loss.detach().item()

        print('--- training epoch %d, lr %f, processed %d/%d, loss %.3f, time elapsed(min) %.2f'
              %(epoch, optimizer.param_groups[-1]['lr'], train_size, train_size, train_loss, (time.time()-start_time)/60))
              
        lr_scheduler.step()

    pred_list = []
    label_list = []
    with torch.no_grad():
        for batchidx, batchdata in enumerate(train_loader):
        
            inputs_rmol = [b.to(device) for b in batchdata[:rmol_max_cnt]]

            labels = (batchdata[-1] - train_y_mean) / train_y_std
            labels = labels.to(device)
            
            pred_ = net(inputs_rmol)
            pred = (pred_*train_y_std)+train_y_mean
            pred_list.append(pred.cpu().numpy())
            label_list.append(labels.cpu().numpy())

    predictions = np.concatenate(pred_list, axis=0)
    labels_ = np.concatenate(label_list, axis=0)

    mse_train = mean_squared_error(labels_, predictions)

    print('training terminated at epoch %d' %epoch)
    
    return net, mse_train
    

def inference(net, mse_train, test_loader, train_y_mean, train_y_std, cuda = torch.device('cuda:0')):
    batch_size = test_loader.batch_size
    
    try:
        rmol_max_cnt = test_loader.dataset.dataset.rmol_max_cnt
    except:
        rmol_max_cnt = test_loader.dataset.rmol_max_cnt
             
    net.eval()
    
    label_list = []
    pred_list = [] 
    
    with torch.no_grad():
        for batchidx, batchdata in enumerate(test_loader):
            inputs_rmol = [b.to(device = torch.device("cuda" if torch.cuda.is_available() else "cpu")) for b in batchdata[:rmol_max_cnt]]

            labels = (batchdata[-1])
            labels = labels.to(device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   )
            
            pred_ = net(inputs_rmol)
            pred = (pred_*train_y_std)+train_y_mean
            pred_list.append(pred.cpu().numpy())
            label_list.append(labels.cpu().numpy())

    predictions = np.concatenate(pred_list, axis=0)
    labels_ = np.concatenate(label_list, axis=0)

    mse = mean_squared_error(labels_, predictions)
    mae = mean_absolute_error(labels_, predictions)
    r2 = r2_score(labels_, predictions)
    rmse = np.sqrt(mse)

    predictions = [element for innerList in predictions for element in innerList]
    pred = torch.Tensor(predictions)
    lab = torch.Tensor(labels_)
    var = torch.tensor(mse_train)
    
    nll_loss = torch.nn.GaussianNLLLoss()
    nlpd = nll_loss(lab, pred, var)
   
    print("RMSE: ", rmse)
    print("MAE: ", mae)
    print("R2: ", r2)   
    print("NLPD: ", nlpd)
    
    return rmse, mae, r2


import numpy as np
import sys, csv, os
import torch
from torch.utils.data import DataLoader
from dgl.data.utils import split_dataset

from graph_dataset import GraphDataset
from dkl_utils import collate_reaction_graphs

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats

train_size = 3164 #[3164, 2767, 1977, 1186, 791, 395, 197, 98]
batch_size = 128

data = GraphDataset()
frac_split = (train_size + 1e-5)/len(data)
train_set, test_set = split_dataset(data, [frac_split, 1 - frac_split], shuffle=False)

train_loader = DataLoader(dataset=train_set, batch_size=int(np.min([batch_size, len(train_set)])), shuffle=True, collate_fn=collate_reaction_graphs, drop_last=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)

print('-- CONFIGURATIONS')
print('--- train/test: %d/%d' %(len(train_set), len(test_set)))
print('--- max no. reactants:', data.rmol_max_cnt)
print('--- max no. products:', data.pmol_max_cnt)

# training 
train_y = train_loader.dataset.dataset.yld[train_loader.dataset.indices]
train_y_mean = np.mean(train_y)
train_y_std = np.std(train_y)

node_dim = data.rmol_node_attr[0].shape[1]
edge_dim = data.rmol_edge_attr[0].shape[1]
net = reactionMPNN(node_dim, edge_dim)
net, mse_train = training(net, train_loader, None, train_y_mean, train_y_std)

# inference
test_y = test_loader.dataset.dataset.yld[test_loader.dataset.indices]

rmse, mae, r2 = inference(net, mse_train, test_loader, train_y_mean, train_y_std)
