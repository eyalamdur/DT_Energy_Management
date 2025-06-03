import os
import pickle
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from typing import List, Dict, Optional
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


def print_stats(stats_file, step, state, action, reward, done):
    lines = []

    lines.append(f"     step: {step}")

    # Device labels for paired P and Q in state
    device_labels = ["Slack", "Load1", "PV", "Load2", "Wind", "EV", "Storage"]

    lines.append("         state:")
    for i, label in enumerate(device_labels):
        p = state[i]
        q = state[i + 7]
        lines.append(f"             {label:<8} P: {p:>7.3f}   Q: {q:>7.3f}")

    # Additional state values
    lines.append(f"             Storage SoC     : {state[14]:7.3f}")
    lines.append(f"             PV Max          : {state[15]:7.3f}")
    lines.append(f"             Wind Max        : {state[16]:7.3f}")
    lines.append(f"             Time Index      : {int(state[17])}")

    # Actions
    lines.append("         action:")
    lines.append(f"             Slack setpoint     P: {action[0]:7.3f}   Q: {action[1]:7.3f}")
    lines.append(f"             Storage dispatch   P: {action[2]:7.3f}   Q: {action[3]:7.3f}")
    lines.append(f"             PV curtailment     P: {action[4]:7.3f}")
    lines.append(f"             Wind curtailment   P: {action[5]:7.3f}")

    lines.append(f"         reward: {reward:.3f}")
    lines.append(f"         done  : {done}")

    # Write all lines to the file
    stats_file.write("\n".join(lines) + "\n")
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

def get_next_run_dir(base_dir: str, agent_type: str) -> str:
    """
    Get the next available run directory for a specific agent type.
    Args:
        base_dir (str): The base directory for the agent type.
        agent_type (str): The agent type (e.g., "random", "PPO", "TD3").
    Returns:
        str: The path to the next available run directory.
    """
    agent_dir = os.path.join(base_dir, agent_type)
    os.makedirs(agent_dir, exist_ok=True)
    existing_runs = [d for d in os.listdir(agent_dir) if os.path.isdir(os.path.join(agent_dir, d)) and d.startswith('run')]
    indices = [int(d[4:]) for d in existing_runs if d[4:].isdigit()]
    next_index = max(indices) + 1 if indices else 0
    run_dir = os.path.join(agent_dir, f'run_{next_index}')
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_trajectories(trajectories: List[Dict[str, np.ndarray]], agent_type: str) -> None:
    """
    Save trajectories to logs/trajectories/<agent_type>/run(x)/trajectories.pkl
    Args:
        trajectories (List[Dict[str, np.ndarray]]): The collected trajectories.
        agent_type (str): The type of agent (e.g., "random", "PPO", "TD3").
    """
    run_dir = get_next_run_dir("logs/trajectories", agent_type)
    with open(os.path.join(run_dir, "trajectories.pkl"), "wb") as f:
        pickle.dump(trajectories, f)
    print(f"[✓] Saved {agent_type} trajectories to {run_dir}")

def save_model(model: nn.Module, agent_type: str) -> None:
    """
    Save model to logs/dt_models/<agent_type>/model(x)/model.pt
    Args:
        model (nn.Module): The model to save.
        agent_type (str): The type of agent (e.g., "random", "PPO", "TD3").
    """
    run_dir = get_next_run_dir("logs/dt_models", agent_type)
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    print(f"[✓] Saved {agent_type} DT model to {run_dir}")

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
    
