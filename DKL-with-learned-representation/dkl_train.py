import sys, os
import torch

from dkl_utils import evaluate_dkl_model
from dkl import DKLModel


# torch.manual_seed(42); np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

model = DKLModel(gp_kernel='matern', use_ard=False, use_lengthscale_prior=False).to(device)

print('-- CONFIGURATIONS')
print(f"\tNum parameters {sum(p.numel() for p in model.parameters())}")
print("device", device)

evaluate_dkl_model(model)

