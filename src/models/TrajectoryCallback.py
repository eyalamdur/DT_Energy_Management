import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# Callback to collect trajectories from the rollout buffer
class TrajectoryCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.trajectories = []

    def _on_step(self) -> bool:
        # Required abstract method; return True to continue training
        return True

    def _on_rollout_end(self) -> None:
        buf = self.model.rollout_buffer
        states = buf.observations.copy()
        actions = buf.actions.copy()
        rewards = buf.rewards.copy()
        dones = buf.dones.copy()

        # Truncate at episode end if done
        if dones.any():
            end_idx = int(np.where(dones)[0][0]) + 1
            states = states[:end_idx]
            actions = actions[:end_idx]
            rewards = rewards[:end_idx]

        # Compute return-to-go
        rtgs = np.cumsum(rewards[::-1])[::-1]

        self.trajectories.append({
            "states": states,
            "actions": actions,
            "rtgs": rtgs
        })
        buf.reset()
