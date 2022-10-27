import copy
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
# plt.style.use('ggplot')


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int = 128, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x.permute(1,0,2)
        x = self.dropout(x + self.pe[:x.size(0)])
        x = x.permute(1,0,2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead):
        super(EncoderLayer, self).__init__()

        self.feature_map = elu_feature_map

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.attention = MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # !!!!!! elu ?
        query = self.q_proj(query)  # [N, L, C]
        key = self.k_proj(key)  # [N, S, C]
        value = self.v_proj(value)

        message, attn_weights = self.attention(query, key, value)  
        message = self.merge(message)  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message, attn_weights


class TransformerEncoder(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, input_dim=3, d_model=128, n_head=8):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.nhead = n_head
        self.fc1 = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layer_names = ['self', 'cross']
        # !!!!!!!!!
        encoder_layer = EncoderLayer(self.d_model, self.nhead)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self.fc2 = nn.Linear(d_model, 1)
        # self.w = nn.parameter.Parameter(torch.rand(d_model,d_model))
        # self._reset_parameters()


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



    def forward(self, dense_map, sparse_map, sparse_index, mask0=None, mask1=None):
        """
        Args:
            dense_map (torch.Tensor): [N, L, C]
            sparse_map (torch.Tensor): [N, S, C]
            sparse_index (torch.Tensor) [N]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        # position encoding
        dense_map = self.pos_encoding(self.fc1(dense_map))
        sparse_map = self.pos_encoding(self.fc1(sparse_map))

        assert self.d_model == dense_map.size(2), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                dense_map, atten0_self = layer(dense_map, dense_map, mask0, mask0)
                sparse_map, atten1_self = layer(sparse_map, sparse_map, mask1, mask1)

                dense_map_self_atten = dense_map
                spare_map_self_atten = sparse_map


            elif name == 'cross':
                sparse_map = torch.stack([sparse_map[i,j,:] for i,j in enumerate(sparse_index)], dim=0).unsqueeze(1)
                dense_map, atten0_cross = layer(dense_map, sparse_map, mask0, mask1)
            else:
                raise KeyError


        logits = self.fc2(dense_map).squeeze(-1)
        # logits = logits - torch.max(logits,1)[0][:,None]
        return logits, dense_map_self_atten, spare_map_self_atten



if __name__ == '__main__':
    net = TransformerEncoder()
    dense_map = torch.zeros((2,441,3), dtype=torch.float32)
    sparse_map = torch.zeros((2,441,3), dtype=torch.float32)
    sparse_index = torch.tensor([0,0])
    out = net(dense_map, sparse_map, sparse_index)
    print(out.shape)




