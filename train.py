import torch
from torch.utils.data import DataLoader
import time
import math
import argparse
from model.full_model import Model
import pdb
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import datetime
import os
import numpy as np
import random
import torchvision.transforms as tsfm
inv_normalize = tsfm.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])


# Initialize device:

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', default="eth")
parser.add_argument('--dataset', default="trajnet")
parser.add_argument('--device', default="cuda:0")
parser.add_argument('--add-noise', default="False")
args = parser.parse_args()

# Seed
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device(args.device)
dataset = args.dataset
add_noise = args.add_noise
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

val_set = DS(dataset, horizon=horizon, fut_len=fut_len, type=val_type, grid_extent=grid_extent)

tr_dl = DataLoader(tr_set,
                   batch_size=train_batch_size,
                   shuffle=True,
                   num_workers=8
                   )
val_dl = DataLoader(val_set,
                    batch_size=64,
                    shuffle=False,
                    num_workers=8
                    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


net = Model(horizon, fut_len, nei_dim, grid_extent).float().to(device)

temp = 1

parameters = list(net.parameters())

t_optimizer = torch.optim.Adam(parameters, lr=1e-3)  #

st_time = time.time()

start_epoch = 1
min_val_loss = math.inf
min_epoch = 0

start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
training_result_dir = "./results/" + start_time + "/"
if not os.path.exists(training_result_dir):
    os.makedirs(training_result_dir)



type = "omgs"
net.train()
totali = 0
for epoch in range(step4):

    if epoch == step1:

        checkpoint = torch.load(dataset + "omgs.tar")

        net.load_state_dict(checkpoint['model_state_dict'])

        type = "dist"
        t_optimizer = torch.optim.Adam(parameters, lr=1e-3)
        min_val_loss = math.inf

    elif epoch == step2:

        checkpoint = torch.load(dataset + "dist.tar")

        net.load_state_dict(checkpoint['model_state_dict'])

        temp = checkpoint['temp']
        min_val_loss = math.inf
        type = "cluster"
        t_optimizer = torch.optim.Adam(parameters, lr=1e-3)  # , weight_decay=1e-4


    elif epoch == step3:

        checkpoint = torch.load(dataset + "cluster.tar")

        net.load_state_dict(checkpoint['model_state_dict'])

        temp = checkpoint['temp']
        type = "end"
        t_optimizer = torch.optim.Adam(parameters, lr=1e-4)  # , weight_decay=1e-4

    for i, data in enumerate(tr_dl):
        totali +=1
        loss, _, _, _, _, _, count = net(data, temp=temp, type=type, device=device, add_noise=add_noise)
        # print("loss: ",loss)
        writer.add_scalar('loss', loss, totali )
        
        
        
        t_optimizer.zero_grad()
        loss.backward()
        a = torch.nn.utils.clip_grad_norm_(parameters, 10)
        t_optimizer.step()


        if type == "dist" :
            temp = temp * gamma

        if i % eval_every == 0:
            net.eval()

            with torch.no_grad():
                agg_val_loss = 0
                policy_loss = 0
                traj_loss = 0
                ogms_rce_loss = 0
                ogms_ce_loss = 0
                l_batch_loss = 0
                val_batch_count = 0

                # Load batch
                for k, data_val in enumerate(val_dl):
                    loss, policy, traj, ogms_rce, ogms_ce, min_ade, count = net(data_val, temp=temp, type=type,
                                                                                device=device, add_noise = add_noise)

                    agg_val_loss += loss.item() * count
                    policy_loss += policy.item()
                    traj_loss += traj.item()
                    ogms_rce_loss += ogms_rce.item()
                    ogms_ce_loss += ogms_ce.item()
                    l_batch_loss += min_ade.item() * count
                    val_batch_count += count

            val_loss = agg_val_loss / val_batch_count
            

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                min_epoch = epoch
                model_path = dataset + type + '.tar'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'temp': temp,
                    'loss': val_loss,
                    'min_val_loss': min_val_loss
                }, os.path.join(training_result_dir,model_path))

            print("Epoch no:", epoch,
                  "| temp", format(temp, '0.5f'),
                  "| val", format(agg_val_loss / val_batch_count, '0.3f'),
                  "| ade", format(l_batch_loss / val_batch_count, '0.3f'),
                  "| policy", format(policy_loss / val_batch_count, '0.3f'),
                  "| traj", format(traj_loss / val_batch_count, '0.3f'),
                  "|ce", format((traj_loss + policy_loss) / val_batch_count, '0.3f'),
                  "| ogms_rce", format(ogms_rce_loss / val_batch_count, '0.3f'),
                  "| ogms_ce", format(ogms_ce_loss / val_batch_count, '0.3f'),
                  "| Min epoch", min_epoch,
                  "| Min val loss", format(min_val_loss, '0.3f'),
                  "| T(s):", int(time.time() - st_time))

            st_time = time.time()

            net.train()

    writer.add_scalar('current time', time.time(), epoch )
    writer.add_image('noise', net.noise, epoch)
    writer.add_images('my_image_batch', inv_normalize(data[3]), epoch)
    writer.add_image('sample a img', inv_normalize(data[3][0]), epoch)