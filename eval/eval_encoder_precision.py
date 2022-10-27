import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import gym
import time
import os
import matplotlib.pyplot as plt
import random

import sys 
sys.path.append('../..')
import gymEnv
from utils.logx import EpochLogger
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from utils.util import read_force_2d_map, cosine_similarity, CustomDataset, read_dense_map
from baselines.algo6.transformer_1 import TransformerEncoder




def collect_data(shapes, steps_per_epoch, n_env, seed):
    dense_map_buf = np.zeros([steps_per_epoch, 441, 6], dtype=np.float32)
    sparse_map_buf = np.zeros([steps_per_epoch, 441, 6], dtype=np.float32)
    dense_index_buf = np.zeros([steps_per_epoch], dtype=np.long)
    sparse_index_buf = np.zeros([steps_per_epoch], dtype=np.long)
    ep_len_buf = np.zeros([steps_per_epoch], dtype=np.long)
    
    assert steps_per_epoch % n_env == 0, "n_env must be divided by steps_per_epoch"

    trials_ls = np.zeros((21),dtype=np.int)

    size = steps_per_epoch // n_env
    for n in range(n_env):
        # collecting with different dense maps
        shape = shapes[n%(len(shapes))]
        random_alpha = random.choice(np.arange(0, alpha*10, alpha)/10)
        dense_map = read_force_2d_map(map_path, shape, pooling=pooling, alpha=random_alpha, seed=seed+n)
        env = gym.make('gymEnv:peg-in-hole-v4', dense_map=dense_map, beta=beta, seed=seed+n)

        _, dense_map, sparse_map, dense_index, sparse_index = env.reset()

        epLen = 0

        for i in range(size):
            # print('n/i', n, i)
            epLen += 1
            dense_map_buf[n*size+i] = dense_map
            sparse_map_buf[n*size+i] = sparse_map
            dense_index_buf[n*size+i] = dense_index
            sparse_index_buf[n*size+i] = sparse_index
            ep_len_buf[n*size+i] = epLen
            
            a = np.random.uniform(-2.0, 2.0, size=2)*0.3
            # print('a', a)

            _, dense_map, sparse_map, dense_index, sparse_index, r, d = env.step(a)

            trials_ls[epLen-1] += 1

            if d:
                # print('epLen', epLen)
                epLen = 0
                _, dense_map, sparse_map, dense_index, sparse_index = env.reset()
    
    return dense_map_buf, sparse_map_buf, dense_index_buf, sparse_index_buf, ep_len_buf





def eval_encoder():
    for epoch in range(epochs):
        dense_map_buf, sparse_map_buf, dense_index_buf, sparse_index_buf, ep_len_buf = collect_data(
            shapes, eval_steps, n_env=n_env, seed=seed+epoch)
        test_set = CustomDataset(dense_map_buf, sparse_map_buf, dense_index_buf, sparse_index_buf, ep_len_buf)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        avg_distance = 0
        distance_per_step = [[] for i in range(21)]

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # mind the order
                dense_map, sparse_map, sparse_index, dense_index, ep_len = batch
                logits, _,_ = encoder(dense_map.to(device), sparse_map.to(device), sparse_index.to(device))
                label = dense_index.to(device)


                pred = logits.max(dim=-1)[1].cpu().numpy()

                dx_pred, dy_pred = pred // 21, pred % 21

                dx_gt, dy_gt = dense_index.cpu().numpy() // 21, dense_index.cpu().numpy() % 21
                distance = np.sqrt((dx_pred-dx_gt)**2+(dy_pred-dy_gt)**2)
                # distance_ls.append(distance)
                avg_distance += np.mean(distance)

                [distance_per_step[ep_len[i]-1].append(distance[i]) for i in range(len(distance))]

        writer.add_scalar('avg_distance', avg_distance/len(test_loader), epoch)

        for i in range(21):
            writer.add_scalar('distance/'+str(i), np.mean(distance_per_step[i]), epoch)
            writer.add_scalar('std/'+str(i), np.std(distance_per_step[i]), epoch)










if __name__ == '__main__':
    root = '/home/xieliang/models/demos'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=10000)
    parser.add_argument('--act_dim', type=int, default=2)

    parser.add_argument('--map_path', type=str, 
        default=os.path.join(root, 'map/0717'))
    parser.add_argument('--model_path', type=str,
        default=os.path.join(root, 'encoder/1001_multi_shape/model_state_dict.pt'))
    parser.add_argument('--test_path', type=str,
        default=os.path.join(root, 'encoder/1001_multi_shape/hexagon'))

    # !!!!!!!!!!
    parser.add_argument('--alpha', type=float, default=2.0)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--pooling', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_env', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--input_dim', type=int, default=6)
    args = parser.parse_args()



    # !!!!!!!!!
    # shapes = ['square']
    # shapes = ['triangle','pentagon','hexagon', 'trapezoid', 'diamond',
    #     'fillet-1', 'fillet-2', 'fillet-3', 'convex-1', 'convex-2', 
    #     'concave', 'cross']
    shapes = ['hexagon']


    encoder = TransformerEncoder(input_dim=args.input_dim,
        d_model=args.d_model, n_head=args.n_head).to(args.device)
    encoder.load_state_dict(
        torch.load(args.model_path, map_location=args.device))
    writer = SummaryWriter(args.test_path)


    criterion = nn.CrossEntropyLoss()
    device = args.device
    eval_steps = args.eval_steps
    model_path = args.model_path
    map_path = args.map_path
    pooling = args.pooling
    seed = args.seed
    alpha = args.alpha
    beta = args.beta
    batch_size = args.batch_size
    epochs =args.epochs
    n_env = args.n_env

    torch.manual_seed(seed)
    np.random.seed(seed)

    eval_encoder()






