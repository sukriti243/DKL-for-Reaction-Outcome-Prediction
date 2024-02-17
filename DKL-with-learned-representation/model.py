# https://github.com/seokhokang/reaction_yield_nn/blob/main/model.py

import torch
import gpytorch
import torch.nn as nn
from dgl.nn.pytorch import NNConv, Set2Set


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


class GraphFeatureExtractor(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats):
        
        super(GraphFeatureExtractor, self).__init__()

        self.mpnn = MPNN(node_in_feats, edge_in_feats)
    
    def forward(self, rmols):

        r_graph_feats = torch.sum(torch.stack([self.mpnn(mol) for mol in rmols]), 0)

        return r_graph_feats
    

class ExactGPLayer(gpytorch.models.ExactGP):
    '''
    Parameters learned by the model:
        likelihood.noise_covar.raw_noise
        covar_module.raw_outputscale
        covar_module.base_kernel.raw_lengthscale
    '''
    def __init__(self, train_x, train_y, likelihood, kernel, ard_num_dims=None, use_lengthscale_prior=True):
       
        super().__init__(train_x, train_y, likelihood)
        # self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = gpytorch.means.ZeroMean()

        ## Linear kernel
        if kernel=='linear':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        ## RBF kernel
        elif kernel=='rbf' or kernel=='RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims))
        ## Matern kernel (52)
        elif kernel=='matern':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims=ard_num_dims))
        else:
            raise ValueError("[ERROR] the kernel '" + str(kernel) + "' is not supported!")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

