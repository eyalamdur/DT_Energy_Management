import numpy as np
import gymnasium as gym
from typing import List, Dict, Optional, Union
from stable_baselines3.common.base_class import BaseAlgorithm  # parent of PPO, TD3, etc.

# -------------------------------------------- Environment Utilities -------------------------------------------- #
def create_environment(env_name : str, entry_point : str) -> gym.Env:
    """
    Create a gym environment with the specified name and maximum episode steps.
    Args:
        env_name (str): The name of the environment to create.
        entry_point (str): The entry point for the environment.
    Returns:
        env (gym.Env): The created gym environment.
    """
    from gymnasium.envs.registration import registry
    # Check if env is already registered
    if env_name not in [spec.id for spec in registry.values()]:
        gym.register(id=env_name, entry_point=entry_point)

    # Create and return the environment
    return gym.make(env_name)

def collect_trajectories(
    env: gym.Env,
    model: Optional[BaseAlgorithm] = None,
    num_episodes: int = 10,
    min_traj_length: int = 1,
    max_traj_length: int = 100,
    deterministic: bool = True
) -> List[Dict[str, np.ndarray]]:
    """
    Collect trajectories using a model or random actions. Stores return-to-go (RTG) instead of rewards.
    Args:
        env (gym.Env): The environment.
        model (BaseAlgorithm or None): PPO, TD3, or None for random actions.
        num_episodes (int): Number of episodes to collect.
        min_traj_length (int): Minimum length to keep a trajectory.
        deterministic (bool): If using model, whether actions are deterministic.
    Returns:
        List[Dict]: Each dict contains 'states', 'actions', 'rtgs' for one episode.
    """
    trajectories = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        states, actions, rewards = [], [], []

        for _ in range(max_traj_length): 
            # Use model to predict action or sample from action space if no model is provided
            action = model.predict(obs, deterministic=deterministic)[0] if model else env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)

            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            obs = next_obs
            
            if terminated or truncated:
                break

        if len(states) >= min_traj_length:
            rtgs = np.cumsum(rewards[::-1])[::-1]  # Return-to-go at each timestep
            trajectories.append({
                "states": np.array(states),
                "actions": np.array(actions),
                "rtgs": rtgs
            })

    return trajectories

# ----------------------------------------------- Model Utilities ----------------------------------------------- #
def is_model_available(model_name: str) -> bool:
    """
    Check if a model is available in the current directory.
    Args:
        model_name (str): The name of the model to check.
    Returns:
        bool: True if the model is available, False otherwise.
    """
    import os
    return os.path.exists(f"{model_name}.zip") or os.path.exists(model_name)

# ----------------------------------------------- Print Utilities ----------------------------------------------- #
def color_print(text: str, color: str = "blue") -> None:
    """
    Print text in a specified color.
    Args:
        text (str): The text to print.
        color (str): The color to print the text in. Options are 'red', 'green', 'blue', 'yellow'.
    """
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "blue": "\033[94m",
        "yellow": "\033[93m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, colors['blue'])}{text}{colors['reset']}")
    
