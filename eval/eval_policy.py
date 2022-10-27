import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import gym
import time
import os
import matplotlib.pyplot as plt
import random

import sys 
sys.path.append('../..')
import gymEnv
from baselines.algo6 import core as core
from utils.logx import EpochLogger
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from utils.util import read_dense_map, cosine_similarity, read_force_2d_map
from spinup.utils.run_utils import setup_logger_kwargs
from baselines.algo6.transformer_1 import TransformerEncoder




def policy(seed=0, episodes=50, rl_path='', d_model=128, test_path='',
    n_head=8, device='cpu', encoder_path='', actor_type='', input_dim=3,
    map_path='', eval_shape='', pooling=True, alpha=1, beta=0.1, mode=''):

    # Random seed
    # seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    writer = SummaryWriter(test_path)

    if mode == 'rl':
        ac = core.ActorCritic(input_dim, actor_type, d_model, n_head, device).to(device)
        ac.load_state_dict(torch.load(rl_path, map_location=device))
        print('rl policy loaded !')
        # !!!!!! reload encoder ?
        # ac.transformerEncoder.load_state_dict(torch.load(encoder_path, map_location=device))
    else:
        encoder = TransformerEncoder(input_dim=input_dim,
            d_model=d_model, n_head=n_head).to(device)
        encoder.load_state_dict(
            torch.load(encoder_path, map_location=device))
        print('random policy loaded !')


    # Main loop: collect experience in env and update/log each epoch
    episode_count = np.zeros(shape=len(eval_shape))

    avg_success = 0
    epLen = 0
    distance_per_step = [[] for i in range(21)]

    cos = nn.CosineSimilarity(dim=-1)

    for episode in range(episodes):
        print('episode', episode)
        shape_idx = episode % len(eval_shape)
        shape = eval_shape[shape_idx]
        random_alpha = random.choice(np.arange(0, alpha*10, alpha)/10)
        dense_map_eval = read_force_2d_map(map_path, shape, pooling=pooling, alpha=random_alpha, seed=seed+episode)

        env_test = gym.make('gymEnv:peg-in-hole-v4', dense_map=dense_map_eval, beta=beta, seed=seed+episode)

        ft, dense_map, sparse_map, dense_index, sparse_index = env_test.reset()

        for i in range(100):
            dense_map = torch.tensor(dense_map, dtype=torch.float32).unsqueeze(0)
            sparse_map = torch.tensor(sparse_map, dtype=torch.float32).unsqueeze(0)
            sparse_index = torch.tensor([sparse_index], dtype=torch.long)
            
            if mode == 'rl':
                a, v, logp, logits, dense_map_self_atten, spare_map_self_atten = ac.step(
                    dense_map.to(device), sparse_map.to(device), sparse_index.to(device))
                if actor_type == 'Categorical':
                    if a == 0:
                        act = np.array([0.2, 0.0])
                    elif a == 1:
                        act = np.array([-0.2, 0.0])
                    elif a == 2:
                        act = np.array([0.0, 0.2])
                    elif a == 3:
                        act = np.array([0.0, -0.2])
                    elif a == 4:
                        act = np.array([0.4, 0.0])
                    elif a == 5:
                        act = np.array([-0.4, 0.0])
                    elif a == 6:
                        act = np.array([0.0, 0.4])
                    elif a == 7:
                        act = np.array([0.0, -0.4])
                elif actor_type == 'Gaussian':
                    act = np.array(a)*0.3
                else:
                    assert KeyError
            else:
                logits, dense_map_self_atten, spare_map_self_atten = encoder(
                    dense_map.to(device), sparse_map.to(device), sparse_index.to(device))
                logits = logits.detach().cpu().numpy()
                dense_map_self_atten = dense_map_self_atten.detach().cpu().numpy()
                spare_map_self_atten = spare_map_self_atten.detach().cpu().numpy()

                pred = logits.argmax(axis=-1)[0]
                dx_pred, dy_pred = pred // 21, pred % 21
                # position projection: grid map -> real world
                # [0,20] -> [-2mm,2mm]
                # act = -np.array([0.2*dx_pred-2, 0.2*dy_pred-2], dtype=np.float32)

                # !!!!!!!!!!!
                act = np.random.uniform(-2.0, 2.0, size=2)*0.3

            # !!!!!!!!!!
            # compute cosine similarity
            # pred = cos(torch.tensor(ft).unsqueeze(0), dense_map[0]).detach().cpu().numpy()

            pred = logits.argmax(axis=-1)
            dx_pred, dy_pred = pred // 21, pred % 21
            dx_gt, dy_gt = dense_index // 21, dense_index % 21
            distance = np.mean(np.sqrt((dx_pred-dx_gt)**2+(dy_pred-dy_gt)**2))
            
            precision = 1 if (abs(dx_pred-dx_gt)<=3 and abs(dy_pred-dy_gt)<=3) else 0
            print('precision', precision)
            if episode==6:
                env_test.visualize(dx_pred, dy_pred, logits, dense_map_self_atten, spare_map_self_atten)

            _, dense_map, sparse_map, dense_index, sparse_index, stepCount, d = env_test.step(act)

            dx_gt, dy_gt = dense_index // 21, dense_index % 21
            if abs(dx_gt-10)<=1 and abs(dy_gt-10)<=1:
                success = 1
                # !!!!!!!!!!
                d = True
                avg_success += 1
                print('success')
            else:
                success = 0

            epLen += 1


            distance_per_step[epLen-1].append(distance)
        

            if d :
                # !!!!!!! the following lines will change the seed
                # dense_map = torch.tensor(dense_map, dtype=torch.float32).unsqueeze(0)
                # sparse_map = torch.tensor(sparse_map, dtype=torch.float32).unsqueeze(0)
                # sparse_index = torch.tensor([sparse_index], dtype=torch.long)
                # if mode == 'rl':
                #     a, v, logp, logits = ac.step(
                #         dense_map.to(device), sparse_map.to(device), sparse_index.to(device))
                # else:
                #     logits = encoder(
                #         dense_map.to(device), sparse_map.to(device), sparse_index.to(device))

                # pred = logits.argmax(axis=-1).cpu().numpy()
                # dx_pred, dy_pred = pred // 21, pred % 21
                # env_test.visualize(dx_pred, dy_pred)

                writer.add_scalar('EvalPolicy/EvalAvgEpSuccess', success, episode)
                writer.add_scalar('EvalPolicy/EvalAvgEpLen', epLen, episode)
                
                episode_count[shape_idx] += 1
                writer.add_scalar('EvalPolicy/'+shape, success, episode_count[shape_idx])

                if success:
                    for i in range(epLen, 21):
                        distance_per_step[i].append(0)

                for i in range(21):
                    if distance_per_step[i]:
                        writer.add_scalar('distance/'+str(i), np.mean(distance_per_step[i]), episode)
                        writer.add_scalar('std/'+str(i), np.std(distance_per_step[i]), episode)

                epLen = 0

                break

    print('avg success', avg_success/episode_count[0])






if __name__ == '__main__':
    # !!!!!!!
    root = '/home/xieliang/models/demos'
    # root = '/home/ur/xiel/models/demos'

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--map_path', type=str, 
        default=os.path.join(root, 'map/0717'))
    parser.add_argument('--encoder_path', type=str,
        default=os.path.join(root, 'encoder/0825_1_random_alpha_random_beta_resume/model_state_dict.pt'))
    parser.add_argument('--rl_path', type=str,
        default=os.path.join(root, 'rl/0902_iter2(e2e)/model_state_dict.pt'))
    parser.add_argument('--test_path', type=str,
        default=os.path.join(root, 'eval/0912/10-step/e2e_trans_rl/test'))

    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--alpha', type=float, default=2)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--pooling', type=bool, default=True)
    # !!!!!!! results from cpu and cuda are slightly different
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--actor_type', type=str, default='Categorical')
    parser.add_argument('--input_dim', type=int, default=6)
    # parser.add_argument('--policy_mode', type=str, default='recursive')
    parser.add_argument('--policy_mode', type=str, default='rl')
    args = parser.parse_args()


    shapes = ['triangle']
    # shapes = ['triangle','pentagon','hexagon', 'trapezoid', 'diamond',
    #     'fillet-1', 'fillet-2', 'fillet-3', 'convex-1', 'convex-2', 
    #     'concave', 'cross']


    policy(seed=args.seed, episodes=args.episodes, rl_path=args.rl_path,
        d_model=args.d_model, n_head=args.n_head, device=args.device,
        encoder_path=args.encoder_path, actor_type=args.actor_type, 
        input_dim=args.input_dim, map_path=args.map_path, 
        eval_shape=shapes, pooling=args.pooling, test_path=args.test_path,
        alpha=args.alpha, beta=args.beta, mode=args.policy_mode)







