import os
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
import skimage
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor



def paramact(linout, ws, bs, phis):
    linoutx = linout.unsqueeze(-1).repeat_interleave(ws.shape[0], dim=2)
    wsx = ws.expand(output.shape[0], output.shape[1], -1)
    bsx = bs.expand(output.shape[0], output.shape[1], -1)
    phisx = phis.expand(output.shape[0], output.shape[1], -1)
    temp = bsx + (torch.sin((wsx*linoutx)+phisx))
    return torch.sum(temp, 2)

nf = 5
samples = 2
in_features = 10
out_features = 3
m = nn.Linear(in_features, out_features, bias=True)
input = torch.randn(1, samples, in_features)
output = m(input)
print(f'output:{output.shape}')

# coefs = nn.Parameter(2*torch.ones(out_features,))
# coefs = nn.Parameter(tensor([0, 1, 2, 3, 4], dtype=torch.float32).reshape(1,1,-1))
ws = nn.Parameter(torch.ones(nf))
bs = nn.Parameter(torch.ones(nf))
phis = nn.Parameter(torch.zeros(nf))
# exws = ws.expand(output.shape[0], output.shape[1], -1)
# print(f'cooef_shape:{coefs.shape}')
# print(f'exws:{exws.size()}')

linoutx = output.unsqueeze(-1).repeat_interleave(ws.shape[0], dim=3)
wsx = ws.expand(output.shape[0], output.shape[1], output.shape[2], -1)
bsx = bs.expand(output.shape[0], output.shape[1], output.shape[2], -1)
phisx = phis.expand(output.shape[0], output.shape[1], output.shape[2], -1)

print(f'ourputsize:{linoutx.size()}')
print(f'wsx:{wsx.size()}')
# print(f'output:{linoutx.size()}')



temp = bsx + (torch.sin((wsx*linoutx)+phisx))

res = torch.sum(temp, 3)

print(f'linear:{linoutx.size()}')
print(f'activation size:{res.shape}')
# res2 = paramact(output, ws, bs, phis)
# res = paramact(output, ws, bs, phis)

# print(f'res_shape:{res.shape}')
# print(f'output:{output}')
# print(f'res:{res}')
