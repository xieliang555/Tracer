import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

import sys
sys.path.append('../..')

# !!!!!!!!!!!!!!
from baselines.algo6.transformer_1 import TransformerEncoder


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class CNN(nn.Module):
    """docstring for CNN"""
    def __init__(self, out_dim):
        super(CNN, self).__init__()
        self.out_dim = out_dim

        self.features = nn.Sequential(
            nn.Conv2d(13,32,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32,32,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32,32,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(32*2*2, 64),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(64,64),
            nn.ReLU(),

            nn.Linear(64,out_dim),
            )

    
    def forward(self, dense_map, sparse_map, logits):
        '''
        Args:
            dense_map (torch.Tensor): [N, 441, 6]
            sparse_map (torch.Tensor): [N, 441, 6]
            sparse_index (torch.Tensor) [N]
        '''
        dense_map = (dense_map-dense_map.min())/(dense_map.max()-dense_map.min())
        sparse_map = (sparse_map-sparse_map.min())/(sparse_map.max()-sparse_map.min())
        logits = (logits-logits.min())/(logits.max()-logits.min())
        inputs = torch.cat((dense_map, sparse_map, logits.unsqueeze(-1)), -1)
        inputs = inputs.view(-1,21,21,13).permute(0,3,1,2)
        
        outs = self.features(inputs)
        outs = torch.flatten(outs,1)
        outs = self.classifier(outs)
        return outs


# class MLP(nn.Module):
#     """docstring for MLP"""
#     def __init__(self, out_dim):
#         super(MLP, self).__init__()
#         self.out_dim = out_dim
#         self.softmax = nn.Softmax(dim=-1)

#         self.classifier = nn.Sequential(
#             nn.Linear(21*21*13, 1280),
#             nn.ReLU(),

#             nn.Dropout(0.1),
#             nn.Linear(1280, 640),
#             nn.ReLU(),
            
#             nn.Dropout(0.1),
#             nn.Linear(640, 32),
#             nn.ReLU(),

#             nn.Linear(32, out_dim),
#             )


#     def forward(self, dense_map, sparse_map, logits):
#         '''
#         Args:
#             dense_map (torch.Tensor): [N, 441, 6]
#             sparse_map (torch.Tensor): [N, 441, 6]
#             logits (torch.Tensor) [N, 441]
#         '''
#         dense_map = (dense_map-dense_map.min())/(dense_map.max()-dense_map.min())
#         sparse_map = (sparse_map-sparse_map.min())/(sparse_map.max()-sparse_map.min())
#         # logits = (logits-logits.min())/(logits.max()-logits.min())
#         logits = self.softmax(logits)
#         inputs = torch.cat((dense_map, sparse_map, logits.unsqueeze(-1)), -1)
#         inputs = inputs.view(-1, 21*21*13)

#         outs = self.classifier(inputs)
#         return outs




class MLP(nn.Module):
    """docstring for MLP"""
    def __init__(self, out_dim):
        super(MLP, self).__init__()
        self.out_dim = out_dim

        self.softmax = nn.Softmax(dim=-1)

        self.classifier = nn.Sequential(
            nn.Linear(21*21, 64),
            nn.ReLU(),

            # nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            
            # nn.Dropout(0.1),
            nn.Linear(64, out_dim),
            )


    def forward(self, dense_map, sparse_map, logits):
        '''
        Args:
            dense_map (torch.Tensor): [N, 441, 6]
            sparse_map (torch.Tensor): [N, 441, 6]
            logits (torch.Tensor) [N, 441]
        '''
        # dense_map = (dense_map-dense_map.min())/(dense_map.max()-dense_map.min())
        # sparse_map = (sparse_map-sparse_map.min())/(sparse_map.max()-sparse_map.min())
        # logits = (logits-logits.min())/(logits.max()-logits.min())
        # inputs = torch.cat((dense_map, sparse_map, logits.unsqueeze(-1)), -1)
        # inputs = inputs.view(-1, 21*21*13)

        logits = self.softmax(logits)
        outs = self.classifier(logits)
        return outs



# class GaussianActor(nn.Module):

#     def __init__(self, act_dim):
#         super().__init__()
        
#         log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
#         self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
#         self.mu_net = CNN(out_dim=act_dim)

#     def _distribution(self, logits):
#         mu = self.mu_net(logits)
#         std = torch.exp(self.log_std)
#         return Normal(mu, std)

#     def _log_prob_from_distribution(self, pi, act):
#         return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


#     def forward(self, logits, act=None):
#         pi = self._distribution(logits)
#         logp_a = None
#         if act is not None:
#             logp_a = self._log_prob_from_distribution(pi, act)
#         return pi, logp_a



class CategoricalActor(nn.Module):
    """docstring for CategoricalActor"""
    def __init__(self, act_dim=8):
        super(CategoricalActor, self).__init__()
        self.logits_net = MLP(out_dim=act_dim)

    def _distribution(self, dense_map, sparse_map, logits):
        logits = self.logits_net(dense_map, sparse_map, logits)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def forward(self, dense_map, sparse_map, logits, act=None):
        pi = self._distribution(dense_map, sparse_map, logits)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a
        


class Critic(nn.Module):

    def __init__(self):
        super().__init__()
        self.v_net = MLP(out_dim=1)

    def forward(self, dense_map, sparse_map, logits):
        return torch.squeeze(self.v_net(dense_map, sparse_map, logits), -1) # Critical to ensure v has right shape.



class ActorCritic(nn.Module):

    def __init__(self, input_dim=3, actor_type='Categorical', d_model=128, n_head=8, device='cuda'):
        super().__init__()

        self.transformerEncoder = TransformerEncoder(input_dim, d_model, n_head)

        # fix the encoder parameters
        for p in self.transformerEncoder.parameters():
            p.requires_grad=False
        print('Encoder fixed !')

        if actor_type == 'Categorical':
            self.pi = CategoricalActor(8)
        elif actor_type == 'Gaussian':
            self.pi = GaussianActor(2)
        else:
            self.pi = None
        self.v  = Critic()


    def step(self, dense_map, sparse_map, sparse_index):
        '''
        Args:
            dense_map (torch.Tensor): [N, 441, 6]
            sparse_map (torch.Tensor): [N, 441, 6]
            sparse_index (torch.Tensor) [N]
        '''
        with torch.no_grad():
            logits, dense_map_self_atten, sparse_map_self_atten = self.transformerEncoder(dense_map, sparse_map, sparse_index)

            pi = self.pi._distribution(dense_map, sparse_map, logits)
            # In collection, batch_size =1
            a = pi.sample()[0]
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(dense_map, sparse_map, logits)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy(), logits.cpu().numpy(), dense_map_self_atten.cpu().numpy(), sparse_map_self_atten.cpu().numpy()

    def act(self, dense_map, sparse_map, sparse_index):
        return self.step(dense_map, sparse_map, sparse_index)[0]



if __name__ == '__main__':
    np.random.seed(0)
    ac = ActorCritic(input_dim=6)
    dense_map = torch.randn(2, 441, 6)
    sparse_map = torch.randn(2, 441, 6)
    sparse_index = torch.tensor([0, 3])
    a, v, lop_a, logits = ac.step(dense_map, sparse_map, sparse_index)
    print(a)
    print(v)
    







