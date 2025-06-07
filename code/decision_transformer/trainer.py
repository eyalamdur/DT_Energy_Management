import gymnasium as gym
import numpy as np
import torch
import time

   
class Trainer:
    def __init__(self, model, optimizer = None, batch_size = 64, loss_fn = torch.nn.MSELoss(), lr = 1e-4):
        self.model = model
        self.lr = lr
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.start_time = time.time()

    def get_batch(self, trajectories: list, seq_max_len: int = 20, min_traj_length: int = 20):
        """
        Organize the given trajectories into a batches structer.
        Args:
            trajectories (list) : examples of given runs in the environment
            seq_max_len (int) : the maximum length of sequences
            min_traj_length (int): minimum size of a trajectory
        Returns:
        """
        # picking only long-enough trajectories
        valid_trajectories = [t for t in trajectories if len(t["states"]) >= min_traj_length]
        if len(valid_trajectories) == 0:
            raise ValueError("No valid trajectories with the required sequence length.")

        # Initialize batches arrays
        state_dim = valid_trajectories[0]['states'].shape[1]
        action_dim = valid_trajectories[0]['actions'].shape[1]

        state_batch = np.zeros((self.batch_size, seq_max_len, state_dim))
        act_batch = np.zeros((self.batch_size, seq_max_len, action_dim))
        rtg_batch = np.zeros((self.batch_size, seq_max_len))
        timestep_batch = np.zeros((self.batch_size, seq_max_len), dtype=int)
        mask_batch = np.zeros((self.batch_size, seq_max_len), dtype=int)  # In case of padding

        for i in range(self.batch_size):
            traj = np.random.choice(valid_trajectories)
            trajectory_len = len(traj["states"])

            # Checking if need to pad the batch with 0-s
            if trajectory_len < seq_max_len:
                si = 0
            else:
                si = np.random.randint(0, trajectory_len - seq_max_len + 1)

            actual_seq_len = min(seq_max_len, trajectory_len - si)

            state_seq = traj["states"][si:si + actual_seq_len]
            act_seq = traj["actions"][si:si + actual_seq_len]
            rtg_seq = traj["rtgs"][si:si + actual_seq_len]
            timestep_seq = np.arange(si, si + actual_seq_len)

            # Fill data
            state_batch[i, :actual_seq_len] = state_seq
            # normalize actions
            act_batch[i, :actual_seq_len] = \
                2 * (act_seq - self.model.action_min) / (self.model.action_max - self.model.action_min) - 1
            rtg_batch[i, :actual_seq_len] = rtg_seq
            timestep_batch[i, :actual_seq_len] = timestep_seq

            # Fill mask: 1 where real, 0 where padded
            mask_batch[i, :actual_seq_len] = 1

        return (
            torch.tensor(state_batch, dtype=torch.float32),
            torch.tensor(act_batch, dtype=torch.float32),
            torch.tensor(rtg_batch, dtype=torch.float32),
            torch.tensor(timestep_batch, dtype=torch.long),
            torch.tensor(mask_batch, dtype=torch.long)  # return the mask too
        )
        
    def train_step(self, states, actions, rtgs, timesteps, mask):
        """
        Train the model using the collected trajectories.
        Args:
            states (torch.Tensor): The states of the environment.
            actions (torch.Tensor): The actions taken in the environment.
            rtgs (torch.Tensor): The reward-to-go values.
            timesteps (torch.Tensor): The time steps in the episode.
            mask (torch.Tensor): The mask for padding.
        Returns:
            loss (float): The loss value after the training step.
        """
        
        action_target = torch.clone(actions)

        # Call forward pass
        _, action_preds, _ = self.model.forward(states, actions, rtgs, timesteps, mask)

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[mask.reshape(-1) > 0]

        loss = self.loss_fn(action_preds, action_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        return loss.item()
    
    def train(self, trajectories: list, epochs: int = 1001):
        """
        Train the model for a specified number of epochs.
        Args:
            epochs (int): The number of epochs to train for.
        """
        states, actions, rtgs, timesteps, mask = self.get_batch(trajectories)
        for epoch in range(epochs):
            loss = self.train_step(states, actions, rtgs, timesteps, mask)
            if epoch % 100 == 0:
                print(f"Epoch: {epoch} - Loss: {loss:.4f}")
