import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import GPT2Model, GPT2Config


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, rtg_dim: int = 1, embed_dim: int = 128,
                 n_layer: int = 8, n_head: int = 8, max_episode_len: int = 1024, seq_len: int = 20):
        """
        C'tor for the DecisionTransformer class.
        Args:
            state_dim(int): number of parameters to describe a state
            act_dim(int): number of parameters to describe an action
            rtg_dim(int): number of parameters to describe a reward-to-go
            embed_dim(int): size of embedded vectors
            n_layer(int): number of transformer layers
            n_head(int): number of attention layers
            max_episode_len(int): maximum size of an episode (for timestamps)
            seq_len(int): size of sequences from the batch
        """
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.rtg_dim = rtg_dim
        self.embed_dim = embed_dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.max_episode_len = max_episode_len
        self.seq_len = seq_len

        config = transformers.GPT2Config(n_embd = self.embed_dim,n_layer = self.n_layer,n_head = self.n_head)

        self.transformer = GPT2Model(config)
        self.state_embed = nn.Linear(state_dim, embed_dim)
        self.action_embed = nn.Embedding(act_dim, embed_dim)
        self.rtg_embed = nn.Linear(rtg_dim, embed_dim)
        self.timestep_embed = nn.Embedding(max_episode_len, embed_dim)

        self.predict_action = nn.Linear(embed_dim, act_dim)
