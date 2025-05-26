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
        
        # Normalization parameters
        self.action_min = None
        self.action_max = None

        # We must use embed_dim=n_heads*N in order to break the input into tokens
        assert embed_dim % n_head == 0, f"embed_dim must be divisible by num_heads (got embed_dim: {embed_dim} and num_heads: {n_head})"

        config = transformers.GPT2Config(n_embd=self.embed_dim, n_layer=self.n_layer, n_head=self.n_head)

        self.embed_ln = nn.LayerNorm(embed_dim)
        self.transformer = GPT2Model(config)
        self.state_embed = nn.Linear(state_dim, embed_dim)
        self.action_embed = nn.Linear(act_dim, embed_dim)
        self.rtg_embed = nn.Linear(rtg_dim, embed_dim)
        self.timestep_embed = nn.Embedding(max_episode_len, embed_dim)
        self.predict_action = nn.Linear(embed_dim, act_dim)
        self.predict_state = nn.Linear(embed_dim, state_dim)
        self.predict_return = nn.Linear(embed_dim, rtg_dim)
        
    def forward(self, states: torch.Tensor, actions: torch.Tensor, rtgs: torch.Tensor,
                timesteps: torch.Tensor, mask: torch.tensor = None) -> torch.Tensor:
        """
        Forward pass of the Decision Transformer.
        Args:
            states (torch.Tensor): states of the environment
            actions (torch.Tensor): normalized actions taken in the environment
            rtgs (torch.Tensor): reward-to-go values
            timesteps (torch.Tensor): time steps in the episode
            mask (torch.Tensor): mask for padding
        Returns:
            torch.Tensor: predicted normalized actions
        """
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        if mask is None:
            mask = torch.ones((batch_size, seq_len), dtype=torch.long)

        # Embedding the inputs
        state_embeddings = self.state_embed(states)
        action_embeddings = self.action_embed(actions)
        rtg_embeddings = self.rtg_embed(rtgs.unsqueeze(-1))
        timestep_embeddings = self.timestep_embed(timesteps)
        
        # Building the tokens
        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + timestep_embeddings
        action_embeddings = action_embeddings + timestep_embeddings
        rtg_embeddings = rtg_embeddings + timestep_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        stacked_inputs = torch.stack((rtg_embeddings, state_embeddings, action_embeddings),
                                     dim=1).permute(0, 2, 1, 3).reshape(batch_size, 3*self.seq_len, self.embed_dim)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack((mask, mask, mask), dim=1).permute(0, 2, 1).reshape(batch_size, 3*self.seq_len)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs,attention_mask=stacked_attention_mask)
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, self.seq_len, 3, self.embed_dim).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state

        return state_preds, action_preds, return_preds

    def get_action(self, states: torch.Tensor, actions: torch.Tensor, rtgs: torch.Tensor, timesteps: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Get the predicted actions from the model.
        Args:
            states (torch.Tensor): states of the environment
            actions (torch.Tensor): actions taken in the environment
            rtgs (torch.Tensor): reward-to-go values
            timesteps (torch.Tensor): time steps in the episode
        Returns:
            torch.Tensor: predicted actions
        """
        # normalize given actions so it would fit the DT engine
        if self.action_min is not None and self.action_max is not None:
            actions = self.normalize_action(actions)

        _, action_preds, _ = self.forward(states, actions, rtgs, timesteps, mask)

        # unnormalize the action prediction before returning it
        if self.action_min is not None and self.action_max is not None:
            action_preds = self.unnormalize_action(action_preds)

        return action_preds[0,-1]

    def normalize_action(self, action):
        """
        Normalize the given action to fit the DT engine.
        This is min-max normalization, so it will return a value between 0 and 1.
        If the action_min and action_max are not set, it will return the action as is.
        Args:
            action (torch.Tensor): Action tensor to normalize.
        Returns:
            torch.Tensor: Normalized action tensor.
        """
        if (self.action_min is None or self.action_max is None) or self.action_max == self.action_min:
            return action
        return (action - self.action_min) / (self.action_max - self.action_min)

    def unnormalize_action(self, norm_action):
        """
        Unnormalize the given normalized action to its original scale.
        This is min-max normalization, so it will return the original action
        by applying the inverse transformation.
        Args:
            norm_action (torch.Tensor): Normalized action tensor to unnormalize.
        Returns:
            torch.Tensor: Unnormalized action tensor.
        """
        if (self.action_min is None or self.action_max is None) or self.action_max == self.action_min:
            return norm_action
        return norm_action * (self.action_max - self.action_min) + self.action_min