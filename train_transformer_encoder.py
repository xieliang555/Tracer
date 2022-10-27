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
from utils.util import read_dense_map, cosine_similarity, CustomDataset, read_force_2d_map
from baselines.algo6.transformer_1 import TransformerEncoder
from utils.run_utils import setup_logger_kwargs
from baselines.algo6 import core as core




def collect_data(shapes, steps_per_epoch, n_env, seed, input_dim, map_path, pooling, 
    alpha, beta, policy='random', rl_model_path='', device='cpu'):
    dense_map_buf = np.zeros([steps_per_epoch, 441, input_dim], dtype=np.float32)
    sparse_map_buf = np.zeros([steps_per_epoch, 441, input_dim], dtype=np.float32)
    dense_index_buf = np.zeros([steps_per_epoch], dtype=np.long)
    sparse_index_buf = np.zeros([steps_per_epoch], dtype=np.long)
    ep_len_buf = np.zeros([steps_per_epoch], dtype=np.long)
    
    assert steps_per_epoch % n_env == 0, "n_env must be divided by steps_per_epoch"

    if policy == 'rl':
        ac = core.ActorCritic(input_dim, actor_type='Categorical', 
            d_model=16, n_head=4, device=device).to(device)
        ac.load_state_dict(torch.load(rl_model_path, map_location=device))
        print('RL policy loaded !')

        # !!!!!!!!!!!!!! reload encoder ?
        # ac.transformerEncoder.load_state_dict(torch.load(encoder_path, map_location=device))

    size = steps_per_epoch // n_env
    for n in range(n_env):
        # collecting with different dense maps
        shape = shapes[n%(len(shapes))]
        # print(n, shape)
        # random_beta = random.choice(np.arange(0, beta*10, beta)/10)
        # !!!!!!!!!!!!!!!
        random_alpha = 0
        # random_alpha = random.choice(np.arange(0, alpha*10, alpha)/10)
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
            
            if policy == 'random':
                act = np.random.uniform(-2.0, 2.0, size=2)*0.3
            elif policy == 'rl':
                dense_map = torch.tensor(dense_map, dtype=torch.float32).unsqueeze(0)
                sparse_map = torch.tensor(sparse_map, dtype=torch.float32).unsqueeze(0)
                sparse_index = torch.tensor([sparse_index], dtype=torch.long)

                a, v, logp, logits, _, _ = ac.step(
                    dense_map.to(device), sparse_map.to(device), sparse_index.to(device))
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

            # env.visualize()
            # !!!!!!!
            # pred = logits.argmax(axis=-1)
            dx_gt, dy_gt = dense_index // 21, dense_index % 21
            # dx_pred, dy_pred = pred // 21, pred % 21
            # precision = 1 if (abs(dx_pred-dx_gt)<=3 and abs(dy_pred-dy_gt)<=3) else 0 
            # # print(precision)

            _, dense_map, sparse_map, dense_index, sparse_index, stepCount, d = env.step(act)


            if abs(dx_gt-10)<=1 and abs(dy_gt-10)<=1 and policy=='rl':
                d = True

            if d:
                # print('epLen', epLen)
                epLen = 0
                _, dense_map, sparse_map, dense_index, sparse_index = env.reset()

    return dense_map_buf, sparse_map_buf, dense_index_buf, sparse_index_buf, ep_len_buf




def train(epoch, train_loader, device, encoder, criterion, optimizer, writer, logger):
    for batch_idx, batch in enumerate(train_loader):
        dense_map, sparse_map, sparse_index, dense_index, ep_len = batch
        logits, _, _ = encoder(dense_map.to(device), sparse_map.to(device), sparse_index.to(device))
        label = dense_index.to(device)

        # compute weighted loss
        loss_list =[]
        for i in range(len(batch[0])):
            # loss_i = criterion(logits[i].unsqueeze(0),label[i].unsqueeze(0))
            loss_i = (ep_len[i]/21)*criterion(logits[i].unsqueeze(0), label[i].unsqueeze(0))
            loss_list.append(loss_i)
        loss = sum(loss_list)/len(batch[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = logits.max(dim=-1)[1].cpu().numpy()
        dx_pred, dy_pred = pred // 21, pred % 21

        dx_gt, dy_gt = dense_index.cpu().numpy() // 21, dense_index.cpu().numpy() % 21
        distance = np.mean(np.sqrt((dx_pred-dx_gt)**2+(dy_pred-dy_gt)**2))

        precision_list_10mm =  [1 if (abs(dx_pred[i]-dx_gt[i])<=5 and abs(dy_pred[i]-dy_gt[i])<=5) else 0 for i in range(len(dense_index))]
        Precision_10mm = np.mean(precision_list_10mm)
        precision_list_6mm =  [1 if (abs(dx_pred[i]-dx_gt[i])<=3 and abs(dy_pred[i]-dy_gt[i])<=3) else 0 for i in range(len(dense_index))]
        Precision_6mm = np.mean(precision_list_6mm)
        precision_list_2mm =  [1 if (abs(dx_pred[i]-dx_gt[i])<=1 and abs(dy_pred[i]-dy_gt[i])<=1) else 0 for i in range(len(dense_index))]
        Precision_2mm = np.mean(precision_list_2mm)
        # print(precision_list_6mm)

        
        logger.store(Loss=loss.item())
        logger.store(Distance=distance)
        logger.store(Precision_10mm=Precision_10mm)
        logger.store(Precision_6mm=Precision_6mm)
        logger.store(Precision_2mm=Precision_2mm)

        writer.add_scalar('loss/train', loss, epoch*len(train_loader)+batch_idx)
        writer.add_scalar('distance/train', distance, epoch*len(train_loader)+batch_idx)
        writer.add_scalar('Precision_2mm/train', Precision_2mm, epoch*len(train_loader)+batch_idx)
        writer.add_scalar('Precision_6mm/train', Precision_6mm, epoch*len(train_loader)+batch_idx)
        writer.add_scalar('Precision_10mm/train', Precision_10mm, epoch*len(train_loader)+batch_idx)


    logger.log_tabular('Epoch', epoch)
    logger.log_tabular('Loss', average_only=True)
    logger.log_tabular('Distance', average_only=True)
    logger.log_tabular('Precision_10mm', average_only=True)
    logger.log_tabular('Precision_6mm', average_only=True)
    logger.log_tabular('Precision_2mm', average_only=True)
    logger.dump_tabular()



def eval(epoch, test_loader, device, encoder, criterion, writer, logger):
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            dense_map, sparse_map, sparse_index, dense_index, ep_len = batch
            logits, _, _ = encoder(dense_map.to(device), sparse_map.to(device), sparse_index.to(device))
            label = dense_index.to(device)

            # compute weighted loss
            loss_list =[]
            for i in range(len(batch[0])):
                # loss_i = criterion(logits[i].unsqueeze(0), label[i].unsqueeze(0))
                loss_i = (ep_len[i]/21)*criterion(logits[i].unsqueeze(0), label[i].unsqueeze(0))
                loss_list.append(loss_i)
            loss = sum(loss_list)/len(batch[0])

            pred = logits.max(dim=-1)[1].cpu().numpy()
            dx_pred, dy_pred = pred // 21, pred % 21

            dx_gt, dy_gt = dense_index.cpu().numpy() // 21, dense_index.cpu().numpy() % 21
            distance = np.mean(np.sqrt((dx_pred-dx_gt)**2+(dy_pred-dy_gt)**2))

            precision_list_10mm =  [1 if (abs(dx_pred[i]-dx_gt[i])<=5 and abs(dy_pred[i]-dy_gt[i])<=5) else 0 for i in range(len(dense_index))]
            Precision_10mm = np.mean(precision_list_10mm)
            precision_list_6mm =  [1 if (abs(dx_pred[i]-dx_gt[i])<=3 and abs(dy_pred[i]-dy_gt[i])<=3) else 0 for i in range(len(dense_index))]
            Precision_6mm = np.mean(precision_list_6mm)
            precision_list_2mm =  [1 if (abs(dx_pred[i]-dx_gt[i])<=1 and abs(dy_pred[i]-dy_gt[i])<=1) else 0 for i in range(len(dense_index))]
            Precision_2mm = np.mean(precision_list_2mm)
            # print(precision_list_6mm)

            logger.store(Loss=loss.item())
            logger.store(Distance=distance)
            logger.store(Precision_10mm=Precision_10mm)
            logger.store(Precision_6mm=Precision_6mm)
            logger.store(Precision_2mm=Precision_2mm)

            writer.add_scalar('loss/test', loss, epoch*len(test_loader)+batch_idx)
            writer.add_scalar('distance/test', distance, epoch*len(test_loader)+batch_idx)
            writer.add_scalar('Precision_10mm/test', Precision_10mm, epoch*len(test_loader)+batch_idx)
            writer.add_scalar('Precision_6mm/test', Precision_6mm, epoch*len(test_loader)+batch_idx)
            writer.add_scalar('Precision_2mm/test', Precision_2mm, epoch*len(test_loader)+batch_idx)

        logger.log_tabular('Loss', average_only=True)
        logger.log_tabular('Distance', average_only=True)
        logger.log_tabular('Precision_10mm', average_only=True)
        logger.log_tabular('Precision_6mm', average_only=True)
        logger.log_tabular('Precision_2mm', average_only=True)
        logger.dump_tabular()





def train_encoder(input_dim, steps_per_epoch, n_env, seed, map_path, pooling, alpha, beta, batch_size,
    eval_steps, epochs, model_path, device, train_shapes, eval_shapes, d_model, n_head, resume,
    pretrained, exp_name, policy='random', rl_model_path=''):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    encoder = TransformerEncoder(input_dim=input_dim, d_model=d_model, n_head=n_head).to(device)
    if resume:
        encoder.load_state_dict(torch.load(pretrained, map_location=device))
        print('Resume training !!!')
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(encoder.parameters(), lr=1e-4)
    writer = SummaryWriter(model_path)

    logger_kwargs = setup_logger_kwargs(exp_name, seed, data_dir=model_path)
    logger = EpochLogger(**logger_kwargs)


    # prepare training data
    dense_map_buf, sparse_map_buf, dense_index_buf, sparse_index_buf, ep_len_buf = collect_data(
        train_shapes, steps_per_epoch, n_env, seed=seed, input_dim=input_dim, map_path=map_path,
        pooling=pooling, alpha=alpha, beta=beta, policy=policy, rl_model_path=rl_model_path,
        device=device)
    train_set = CustomDataset(dense_map_buf, sparse_map_buf, dense_index_buf, sparse_index_buf, ep_len_buf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # prepare testing data
    dense_map_buf, sparse_map_buf, dense_index_buf, sparse_index_buf, ep_len_buf = collect_data(
        eval_shapes, eval_steps, n_env, seed=seed+1, input_dim=input_dim, map_path=map_path,
        pooling=pooling, alpha=alpha, beta=beta, policy=policy, rl_model_path=rl_model_path,
        device=device)
    test_set = CustomDataset(dense_map_buf, sparse_map_buf, dense_index_buf, sparse_index_buf, ep_len_buf)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        train(epoch, train_loader, device, encoder, criterion, optimizer, writer, logger)
        eval(epoch, test_loader, device, encoder, criterion, writer, logger)


        # Save model
        torch.save(encoder.state_dict(), os.path.join(model_path, 'model_state_dict.pt'))
        print('model saved !')






if __name__ == '__main__':
    # !!!!!!!!!
    root = '/home/xieliang/models/demos'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_path', type=str, 
        default=os.path.join(root, 'map/0717'))
    parser.add_argument('--model_path', type=str,
        default=os.path.join(root, 'eval/adaption/encoder/1-shape'))
    parser.add_argument('--exp_name', type=str, default='encoder')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--resume', type=bool, default=True)
    parser.add_argument('--pretrained', type=str,
        default=os.path.join(root, 'encoder/1_shape/model_state_dict.pt'))
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--policy', type=str, default='random')
    # parser.add_argument('--policy', type=str, default='rl')
    parser.add_argument('--rl_model_path', type=str, 
        default=os.path.join(root, 'rl/0902_iter1/model_state_dict.pt'))



    # !!!!!!!
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--beta', type=float, default=0.)
    parser.add_argument('--pooling', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_env', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--input_dim', type=int, default=6)
    parser.add_argument('--steps_per_epoch', type=int, default=10000)
    parser.add_argument('--eval_steps', type=int, default=10000)
    args = parser.parse_args()

    # train_shapes = ['square','triangle', 'pentagon', 'hexagon', 'trapezoid']
    # eval_shape = ['hexagon']
    # eval_shapes = ['diamond','fillet-1', 'fillet-2', 'fillet-3', 
    #     'convex-1', 'convex-2', 'concave', 'cross']


    shapes = [
        'triangle','pentagon', 'hexagon', 'trapezoid', 
        'diamond','fillet-1', 'fillet-2', 'fillet-3', 
        'convex-1', 'convex-2', 'concave', 'cross']

    
    for shape in shapes:
        model_path = os.path.join(args.model_path, shape)
        train_shapes = [shape]
        eval_shapes = [shape]

        train_encoder(input_dim=args.input_dim, steps_per_epoch=args.steps_per_epoch, n_env=args.n_env, seed=args.seed,
            map_path=args.map_path, pooling=args.pooling, alpha=args.alpha, beta=args.beta, batch_size=args.batch_size,
            eval_steps=args.eval_steps, epochs=args.epochs, model_path=model_path, device=args.device, 
            train_shapes=train_shapes, eval_shapes=eval_shapes, d_model=args.d_model, n_head=args.n_head,
            resume=args.resume, pretrained=args.pretrained, exp_name=args.exp_name, policy=args.policy,
            rl_model_path=args.rl_model_path)






