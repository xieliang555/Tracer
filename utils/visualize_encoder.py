import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import gym
import time
import os
import matplotlib.pyplot as plt

import sys 
sys.path.append('../..')
import gymEnv
from utils.logx import EpochLogger
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from utils.util import read_force_2d_map, cosine_similarity, read_dense_map
from baselines.algo6.transformer_1 import TransformerEncoder





def eval_encoder():

    _, dense_map, sparse_map, dense_index, sparse_index = env.reset()
    epLen = 0
    avg_succ = 0
    avg_distance = 0
    index =  np.zeros((21),dtype=np.int)
    trials = np.zeros((21),dtype=np.int)
    print('reset')

    with torch.no_grad():
        for t in range(1000):
            epLen += 1 

            dense_map = torch.tensor(dense_map, dtype=torch.float32).unsqueeze(0)
            sparse_map = torch.tensor(sparse_map, dtype=torch.float32).unsqueeze(0)
            sparse_index = torch.tensor([sparse_index], dtype=torch.long)
            logits = encoder(dense_map.to(device), sparse_map.to(device), sparse_index.to(device))
            label = torch.tensor([dense_index], dtype=torch.long).to(device)

            ax = plt.subplot(2,3,6)
            ax.set_title('logits')
            logits_show = (logits-logits.min())/(logits.max()-logits.min())
            plt.imshow((logits_show).reshape(21,21))


            loss = criterion(logits, label)
            print('loss', loss.item())

            # get prediction
            pred = logits.max(dim=-1)[1].item()
            dx_pred, dy_pred = pred // 21, pred % 21
            dx_gt, dy_gt = dense_index // 21, dense_index % 21
            distance = np.mean(np.sqrt((dx_pred-dx_gt)**2+(dy_pred-dy_gt)**2))
            success = 1 if (abs(dx_pred-dx_gt) <=3 and abs(dy_pred-dy_gt)<=3) else 0
            print('epLen', epLen)
            print('distance', distance)
            print('success', success)
            print('\n')

            avg_succ += success
            avg_distance += distance
            # exit(0)
            index[epLen-1] += success
            trials[epLen-1] += 1

            env.visualize(dx_pred, dy_pred)


            a = np.random.uniform(-2.0,2.0,size=2)*0.3

            _, dense_map, sparse_map, dense_index, sparse_index, r, d = env.step(a)

            if d:
                epLen = 0
                _, dense_map, sparse_map, dense_index, sparse_index = env.reset()
                print('reset\n')


        print('avg succ', avg_succ/1000)
        print('avg dis', avg_distance/1000)
        print('index', index)
        print('trials', trials)
        print(index/trials)





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    root = '/home/xieliang/models/demos'
    parser.add_argument('--map_path', type=str, 
        default=os.path.join(root, 'map/0717'))
    parser.add_argument('--model_path', type=str,
        default=os.path.join(root, 'encoder/0803_new_net_new_atten/model_state_dict.pt'))
    
    # !!!!!!!
    parser.add_argument('--seed', '-s', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=2.0)
    # !!!!!!!
    parser.add_argument('--beta', type=float, default=0.8)
    parser.add_argument('--pooling', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--input_dim', type=int, default=6)

    args = parser.parse_args()

    seed = args.seed
    device = args.device
    model_path = args.model_path
    pooling = args.pooling
    d_model = args.d_model
    n_head = args.n_head
    map_path = args.map_path
    alpha = args.alpha
    beta = args.beta
    input_dim = args.input_dim

    torch.manual_seed(seed)
    np.random.seed(seed)


    # !!!!!!!
    shape = 'triangle'

    encoder = TransformerEncoder(input_dim=input_dim, d_model=d_model, n_head=n_head).to(device)
    encoder.load_state_dict(torch.load(model_path, map_location=device))
    criterion = nn.CrossEntropyLoss()

    dense_map = read_force_2d_map(map_path, shape, pooling=pooling, alpha=alpha, seed=seed)
    env = gym.make('gymEnv:peg-in-hole-v4', dense_map=dense_map, beta=beta, seed=seed)


    eval_encoder()





