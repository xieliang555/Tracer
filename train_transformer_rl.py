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


class VPGBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, actor_type, size, gamma=0.99, lam=0.95, device='cpu'):

        if actor_type == 'Categorical':
            self.act_buf = np.zeros(size, dtype=np.long)
        elif actor_type == 'Gaussian':
            self.act_buf = np.zeros((size,2), dtype=np.float32)
        else:
            raise KeyError 

        self.dense_map_buf = np.zeros((size, 441, 6), dtype=np.float32)
        self.sparse_map_buf = np.zeros((size, 441, 6), dtype=np.float32)
        self.logits_buf = np.zeros((size, 441), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.device = device

    def store(self, dense_map, sparse_map, logits, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.dense_map_buf[self.ptr] = dense_map
        self.sparse_map_buf[self.ptr] = sparse_map
        self.logits_buf[self.ptr] = logits
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(
            dense_map=self.dense_map_buf,
            sparse_map=self.sparse_map_buf,
            logits=self.logits_buf,
            act=self.act_buf, ret=self.ret_buf,
            adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v).to(self.device) for k,v in data.items()}



def vpg(seed=0, steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, save_freq=10, model_path='', exp_name='',
        d_model=128, n_head=8, device='cpu', encoder_path='', actor_type='', input_dim=3,
        map_path='', train_shape='', eval_shape='', pooling=True, alpha=1, beta=0.1,
        resume=False, pretrained=''):

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger_kwargs = setup_logger_kwargs(exp_name, seed, data_dir=model_path)
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    writer = SummaryWriter(model_path)

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Create actor-critic module
    ac = core.ActorCritic(input_dim, actor_type, d_model, n_head, device).to(device)
    

    if resume:
        ac.load_state_dict(torch.load(pretrained, map_location=device))
        print('resume training !!')

    # !!!!!!!! reload encoder ?
    ac.transformerEncoder.load_state_dict(torch.load(encoder_path, map_location=device))


    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = VPGBuffer(actor_type, local_steps_per_epoch, gamma, lam, device)

    # Set up function for computing VPG policy loss
    def compute_loss_pi(data):
        dense_map, sparse_map, logits = data['dense_map'], data['sparse_map'], data['logits']
        act, adv, logp_old = data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(dense_map, sparse_map, logits, act)
        loss_pi = -(logp * adv).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        dense_map, sparse_map, logits, ret = data['dense_map'], data['sparse_map'], data['logits'], data['ret']
        return ((ac.v(dense_map, sparse_map, logits) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    criterion = nn.CrossEntropyLoss()

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        # Get loss and info values before update
        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with a single step of gradient descent
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        mpi_avg_grads(ac.pi)    # average grads across MPI processes
        pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent = pi_info['kl'], pi_info_old['ent']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))


    def eval(eval_episode, episode_count):

        size = local_steps_per_epoch // len(eval_shape)
        eval_precision_ls = np.zeros((21),dtype=np.int)
        eval_trials_ls = np.zeros((21),dtype=np.int)
        eval_precision_total = []

        for shape_idx in range(len(eval_shape)):
            # !!!!!!!!!!!!1
            random_alpha = 0
            # random_alpha = random.choice(np.arange(0, alpha*10, alpha)/10)
            dense_map_eval = read_force_2d_map(map_path, eval_shape[shape_idx], pooling=pooling, alpha=random_alpha, seed=seed+eval_episode+shape_idx)
            env_test = gym.make('gymEnv:peg-in-hole-v4', dense_map=dense_map_eval, beta=beta, seed=seed+eval_episode+shape_idx)
            ft, dense_map, sparse_map, dense_index, sparse_index = env_test.reset()
            
            ep_ret, ep_len = 0, 0
            for t in range(size):
                dense_map = torch.tensor(dense_map, dtype=torch.float32).unsqueeze(0)
                sparse_map = torch.tensor(sparse_map, dtype=torch.float32).unsqueeze(0)
                sparse_index = torch.tensor([sparse_index], dtype=torch.long)
                a, v, logp, logits, _, _ = ac.step(
                    dense_map.to(device), sparse_map.to(device), sparse_index.to(device))

                pred = logits.argmax(axis=-1)
                dx_pred, dy_pred = pred // 21, pred % 21
                dx_gt, dy_gt = dense_index // 21, dense_index % 21
                distance = np.mean(np.sqrt((dx_pred-dx_gt)**2+(dy_pred-dy_gt)**2))
                precision = 1 if (abs(dx_pred-dx_gt) <=3 and abs(dy_pred-dy_gt)<=3) else 0
                eval_precision_total.append(precision)
                logger.store(Distance_eval=distance)
                logger.store(precision_eval=precision)

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

                _, dense_map, sparse_map, dense_index, sparse_index, stepCount, d = env_test.step(act)

                dx_gt, dy_gt = dense_index // 21, dense_index % 21
                if abs(dx_gt-10)<=1 and abs(dy_gt-10)<=1:
                    r = 1
                    d = True
                else:
                    r = 0

                ep_ret += r
                ep_len += 1

                eval_precision_ls[ep_len-1] += precision
                eval_trials_ls[ep_len-1] += 1

                logger.store(VVals_eval=v)

                timeout = ep_len == max_ep_len
                terminal = d or timeout
            
                if terminal:
                    episode_count[shape_idx] += 1
                    writer.add_scalar('EvalPolicy/'+eval_shape[shape_idx], ep_ret, episode_count[shape_idx])

                    eval_episode += 1
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet_eval=ep_ret, EpLen_eval=ep_len)
                    writer.add_scalar('EvalPolicy/EvalAvgSuccess', ep_ret, eval_episode)
                    writer.add_scalar('EvalPolicy/EvalAvgEpLen', ep_len, eval_episode)
                    writer.add_scalar('EvalPolicy/EvalAvgPrecision', np.mean(eval_precision_total), eval_episode)

                    ft, dense_map, sparse_map, dense_index, sparse_index = env_test.reset()
                    ep_ret, ep_len = 0, 0
                    eval_precision_total = []


        # ratio = eval_precision_ls/(eval_trials_ls+1e-6)
        return eval_precision_ls, eval_trials_ls, eval_episode, episode_count


    # Prepare for interaction with environment
    start_time = time.time()
    ep_ret, ep_len = 0, 0

    # Main loop: collect experience in env and update/log each epoch
    train_episode = 0 
    eval_episode = 0
    episode_count = np.zeros(shape=len(eval_shape))

    for epoch in range(epochs):

        size = local_steps_per_epoch // len(train_shape)

        for shape_idx in range(len(train_shape)):
            train_precision_ls = np.zeros((21),dtype=np.int)
            train_trials_ls = np.zeros((21),dtype=np.int)
            train_precision_total = []

            # !!!!!!!!!!1
            random_alpha = 0
            # random_alpha = random.choice(np.arange(0, alpha*10, alpha)/10)
            dense_map_train = read_force_2d_map(map_path, train_shape[shape_idx], pooling=pooling, alpha=random_alpha, seed=seed+epoch+shape_idx)
            env_train = gym.make('gymEnv:peg-in-hole-v4', dense_map=dense_map_train, beta=beta, seed=seed+epoch+shape_idx)
            ft, dense_map, sparse_map, dense_index, sparse_index = env_train.reset()

            for t in range(size):
                dense_map = torch.tensor(dense_map, dtype=torch.float32).unsqueeze(0)
                sparse_map = torch.tensor(sparse_map, dtype=torch.float32).unsqueeze(0)
                sparse_index = torch.tensor([sparse_index], dtype=torch.long)

                a, v, logp, logits, _, _ = ac.step(
                    dense_map.to(device), sparse_map.to(device), sparse_index.to(device))

                pred = logits.argmax(axis=-1)
                loss = criterion(torch.tensor(logits, dtype=torch.float32), 
                    torch.tensor(dense_index, dtype=torch.long).unsqueeze(0))

                dx_pred, dy_pred = pred // 21, pred % 21
                dx_gt, dy_gt = dense_index // 21, dense_index % 21
                distance = np.mean(np.sqrt((dx_pred-dx_gt)**2+(dy_pred-dy_gt)**2))
                precision = 1 if (abs(dx_pred-dx_gt) <=3 and abs(dy_pred-dy_gt)<=3) else 0
                train_precision_total.append(precision)
                logger.store(Distance=distance)
                logger.store(precision=precision)
                logger.store(Loss=loss.item())

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

                _, dense_map, sparse_map, dense_index, sparse_index, stepCount, d = env_train.step(act)

                dx_gt, dy_gt = dense_index // 21, dense_index % 21
                if abs(dx_gt-10)<=1 and abs(dy_gt-10)<=1:
                    r = 1 - stepCount / 100
                    d = True
                else:
                    r = 0

                ep_ret += r
                ep_len += 1
                
                # print('d', d, 'ep_len', ep_len)	
                if ep_len <22:
                    train_precision_ls[ep_len-1] += precision
                    train_trials_ls[ep_len-1] += 1
                    # print('warning: ep_len>21')

                # save and log
                buf.store(dense_map, sparse_map, logits, a, r, v, logp)
                logger.store(VVals=v)

                timeout = ep_len == max_ep_len
                terminal = d or timeout
                epoch_ended = t==local_steps_per_epoch-1

                if terminal or epoch_ended:
                    if epoch_ended and not(terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        dense_map = torch.tensor(dense_map, dtype=torch.float32).unsqueeze(0)
                        sparse_map = torch.tensor(sparse_map, dtype=torch.float32).unsqueeze(0)
                        sparse_index = torch.tensor([sparse_index], dtype=torch.long)
                        _, v, _, _, _, _ = ac.step(dense_map.to(device), sparse_map.to(device), sparse_index.to(device))
                    else:
                        v = 0
                    buf.finish_path(v)
                    if terminal:
                        train_episode += 1
                        # only save EpRet / EpLen if trajectory finished
                        logger.store(EpRet=ep_ret, EpLen=ep_len)
                        writer.add_scalar('RL/TrainAvgEpRet', ep_ret, train_episode)
                        writer.add_scalar('RL/TrainAvgEpLen', ep_len, train_episode)
                        writer.add_scalar('RL/TrainAvgPrecision', np.mean(train_precision_total), train_episode)

                    ft, dense_map, sparse_map, dense_index, sparse_index = env_train.reset()
                    ep_ret, ep_len = 0, 0
                    train_precision_total = []

        eval_precision_ls, eval_trials_ls, eval_episode, episode_count = eval(
            eval_episode, episode_count)
        train_ratio = train_precision_ls/(train_trials_ls+1e-6)
        eval_ratio = eval_precision_ls/(eval_trials_ls+1e-6)

        for i in range(21):
            writer.add_scalar('train_precision/'+str(i), train_precision_ls[i], epoch)
            writer.add_scalar('train_trial/'+str(i), train_trials_ls[i], epoch)
            writer.add_scalar('train_ratio/'+str(i), train_ratio[i], epoch)

            writer.add_scalar('eval_precision/'+str(i), eval_precision_ls[i], epoch)
            writer.add_scalar('eval_trial/'+str(i), eval_trials_ls[i], epoch)
            writer.add_scalar('eval_ratio/'+str(i), eval_ratio[i], epoch)

        # Perform VPG update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('Distance', with_min_and_max=True)
        logger.log_tabular('precision', average_only=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('Loss', average_only=True)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)

        logger.log_tabular('EpRet_eval', with_min_and_max=True)
        logger.log_tabular('EpLen_eval', with_min_and_max=True)
        logger.log_tabular('VVals_eval', with_min_and_max=True)
        logger.log_tabular('Distance_eval', with_min_and_max=True)
        logger.log_tabular('precision_eval', average_only=True)
        logger.dump_tabular()


        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            torch.save(ac.state_dict(), os.path.join(model_path, 'model_state_dict.pt'))
            print('model saved !')



if __name__ == '__main__':
    # !!!!!!!
    root = '/Users/xieliang/models/demos'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=2)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--save_freq', type=int, default=10)


    parser.add_argument('--map_path', type=str, 
        default=os.path.join(root, 'map/0717'))
    parser.add_argument('--model_path', type=str,
        default=os.path.join(root, 'eval/adaption/rl/1_shape'))
    parser.add_argument('--encoder_path', type=str,
        default=os.path.join(root, 'eval/adaption/encoder/1-shape'))
    parser.add_argument('--resume', type=bool, default=True)
    parser.add_argument('--pretrained',type=str,
        default=os.path.join(root, 'rl/1_shape/model_state_dict.pt'))

    # !!!!!!!!!!
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--beta', type=float, default=0.)
    parser.add_argument('--pooling', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--actor_type', type=str, default='Categorical')
    parser.add_argument('--input_dim', type=int, default=6)
    # parser.add_argument('--train_shape', type=str, default='square')
    args = parser.parse_args()

    # eval_shapes = ['square']
    # eval_shapes = ['triangle','pentagon','hexagon', 'trapezoid', 'diamond',
    #     'fillet-1', 'fillet-2', 'fillet-3', 'convex-1', 'convex-2', 
    #     'concave', 'cross']


    # train_shapes = ['square','triangle', 'pentagon', 'hexagon', 'trapezoid']
    # eval_shapes = ['square', 'triangle', 'pentagon']
    # # eval_shape = ['hexagon']
    # eval_shapes = ['diamond',
    #      'fillet-1', 'fillet-2', 'fillet-3', 'convex-1', 'convex-2', 
    #      'concave', 'cross']

    shapes = [
        'triangle','pentagon', 'hexagon', 'trapezoid', 
        'diamond','fillet-1', 'fillet-2', 'fillet-3', 
        'convex-1', 'convex-2', 'concave', 'cross']

    for shape in shapes:
        model_path = os.path.join(args.model_path, shape)
        encoder_path = os.path.join(args.encoder_path, shape, 'model_state_dict.pt')
        train_shapes = [shape]
        eval_shapes = [shape]

        vpg(gamma=args.gamma, seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
            model_path=model_path, exp_name=args.exp_name, 
            save_freq=args.save_freq, d_model=args.d_model, n_head=args.n_head, device=args.device,
            encoder_path=encoder_path, actor_type=args.actor_type, input_dim=args.input_dim,
            map_path=args.map_path, train_shape=train_shapes, eval_shape=eval_shapes,
            pooling=args.pooling, alpha=args.alpha, beta=args.beta, resume=args.resume,
            pretrained=args.pretrained)



