import torch
from torch.utils.data import DataLoader
import time
import math
import argparse
from model.full_model import Model
import pdb

# Initialize device:

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="trajnet")
parser.add_argument('--device', default="cuda:0")
args = parser.parse_args()


device = torch.device(args.device)
dataset = args.dataset
eval_every=1000
val_type="test"

if dataset == "ind":
    horizon = 30
    fut_len = 30
    grid_extent = 25
    train_batch_size = 16
    nei_dim = 0
    step1 = 50
    step2 = 100
    step3 = 200
    step4 = 800
    gamma = 0.9996

    from cvpr2022.data.IND.inD import inD as DS

else:
    horizon = 20
    fut_len = 12
    grid_extent = 20
    train_batch_size = 32
    nei_dim = 2
    gamma = 0.9996

    # if dataset=="trajnet":#tain 67952 val :2829
    step1 = 15  
    step2 = 25 
    step3 = 50  
    step4 = 100  

    from cvpr2022.data.SDD.sdd import sdd as DS

    if dataset=="sdd":
        val_type="val"


# Initialize datasets:
tr_set = DS(dataset, horizon=horizon, fut_len=fut_len, grid_extent=grid_extent)

tr_dl = DataLoader(tr_set,
                   batch_size=train_batch_size,
                   shuffle=True,
                   num_workers=8
                   )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


net = Model(horizon, fut_len, nei_dim, grid_extent).float().to(device)

temp = 1

parameters = list(net.parameters())

t_optimizer = torch.optim.Adam(parameters, lr=1e-3)


# tr_set.images[0][0] shape: 409x473x3
# tr_set.images[0] len: 60
# tr_set.images[0] len: 8
tr_set.images[0][0].shape
len(tr_set.images)
import matplotlib.pyplot as plt
fig,ax = plt.subplots(figsize=(10,10))
ax.imshow(tr_set.images[5][1])
plt.show()
tr_set.__getitem__(10)
# vel_hist (7,2), 
# neighbors (7,2,25,25),
# fut (12,2),
# img (3,200,200),
# waypts_e (20,2),
# bc_targets (20,5,25,25),
# waypt_lengths_e (int=2),
# r_mat,scale (2,2),
# ref_pos (float=0.038392063),
# ds_id (2,)
import numpy as np
tr_set.__getitem__(10)[8]
tr_set.__getitem__(10)[9].shape
np.where(tr_set.__getitem__(10)[5]!=0)
np.transpose(tr_set.__getitem__(10)[3],(1,2,0)).shape
fig,ax = plt.subplots()
ax.imshow(np.transpose(tr_set.__getitem__(10)[3],(1,2,0)))
ax.imshow(tr_set.__getitem__(10)[5][1][4])
plt.show()
vel_hist = tr_set.__getitem__(10)[0]
fut = tr_set.__getitem__(10)[2]
waypts_e = tr_set.__getitem__(10)[4]
img = tr_set.__getitem__(10)[3]
img = img[None,:]

temp = 1
type = "omgs"
device = torch.device(args.device)
data = next(iter(tr_dl))
vel_hist, neighbors, fut, img, waypts_e, bc_targets, waypt_lengths_e, r_mat, scale, ref_pos,ds_id = data
net(data,temp=temp, type=type, device=device)
net.vis(vel_hist,fut,waypts_e,img)



next(iter(tr_dl))


pdb.set_trace()