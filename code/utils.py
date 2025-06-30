import os
import pickle
from datetime import datetime
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from tqdm import trange
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
    model_name = type(model).__name__ if model is not None else "Random"

    for episode in trange(1, num_episodes + 1, desc=f"Collecting trajectories from agent {model_name}"):
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

def load_trajectories(agent_type: str, traj_id: int, base_dir: str = "results/trajectories"):
    """
    Load trajectories for a given agent type and trajectory ID.
    Args:
        agent_type (str): The type of agent (e.g., "random", "ppo", "td3").
        traj_id (int or str): The trajectory ID to load.
        base_dir (str): Base directory where trajectories are stored.
    Returns:
        List[Dict[str, np.ndarray]]: List of trajectories, or None if not found.
    """
    agent_dir = os.path.join(base_dir, agent_type)
    if not os.path.exists(agent_dir):
        return None
    # Find the file with the correct traj_id
    for fname in os.listdir(agent_dir):
        if fname.startswith(f"traj_{traj_id}_") and fname.endswith(".pkl"):
            file_path = os.path.join(agent_dir, fname)
            with open(file_path, "rb") as f:
                return pickle.load(f)
    return None

def save_trajectories(trajectories: List[Dict[str, np.ndarray]], agent_type: str, env: gym.Env, base_dir: str = "results/trajectories") -> str:
    """
    Save trajectories to results/trajectories/<agent_type>/traj_#...pkl with full metadata in filename.
    Args:
        trajectories (List[Dict[str, np.ndarray]]): List of trajectories to save.
        agent_type (str): The type of agent (e.g., "random", "PPO", "TD3").
        env (gym.Env): The environment used for collecting the trajectories.
        base_dir (str): Base directory to save the trajectories.
    """
    file_path = generate_trajectory_filename(base_dir, agent_type, trajectories, env)

    with open(file_path, "wb") as f:
        pickle.dump(trajectories, f)

    print(f"[✓] Saved {agent_type} trajectories to {file_path}")
    return file_path

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

def save_model(model: torch.nn.Module,
               trajectory_id: str,
               loss_fn_name: str,
               batch_size: int,
               optimizer_name: str,
               embed_dim: int,
               n_heads: int,
               n_layers: int,
               lr: float,
               base_dir: str = "results/models/DT") -> None:
    """
    Save model with metadata including trajectory run ID from trajectory filename.
    """
    file_path = generate_model_filename(base_dir, trajectory_id, loss_fn_name,
                                        batch_size, optimizer_name, embed_dim, n_heads, n_layers, lr)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(model.state_dict(), file_path)

    print(f"[✓] Saved DT model to {file_path}")

# ------------------------------------------------ Path Utilities ----------------------------------------------- #
def get_next_run_id(agent_dir: str, folder: str) -> int:
    """
    Get the next run ID based on existing files in the specified folder.
    This function scans the specified folder for files that match the expected naming convention
    and extracts the run IDs to determine the next available ID.
    Args:
        agent_dir (str): The directory of the agent.
        folder (str): The folder to scan for existing files.
    Raises:
        ValueError: If the folder type is unknown.
    Returns:
        int: The next available run ID.
    """
    # Determine next run ID
    if folder == "trajectories":
        existing = [f for f in os.listdir(agent_dir) if f.startswith("traj_") and f.endswith(".pkl")]
    elif folder == "models":
        existing = [f for f in os.listdir(agent_dir) if f.startswith("model_")]
    else:
        raise ValueError(f"Unknown folder type: {folder}")
    indices = [int(f.split("_")[1]) for f in existing if f.split("_")[1].isdigit()]
    return max(indices, default=-1) + 1

def generate_trajectory_filename(base_dir: str,
                                 agent_type: str,
                                 trajectories: List[Dict[str, np.ndarray]],
                                 env) -> str:
    """
    Generate full filename path for saving trajectories with embedded metadata.
    Args:
        base_dir (str): Base directory to save the trajectories.
        agent_type (str): The type of agent (e.g., "random", "PPO", "TD3").
        trajectories (List[Dict[str, np.ndarray]]): List of trajectories to save.
        env (gym.Env): The environment used for collecting the trajectories.
    Returns:
        str: Full path to the trajectory file.
    """
    # Ensure directory exists
    agent_dir = os.path.join(base_dir, agent_type)
    os.makedirs(agent_dir, exist_ok=True)

    # Find next run ID
    run_id = get_next_run_id(agent_dir, "trajectories")

    # Extract metadata
    lengths = [len(traj["states"]) for traj in trajectories]
    min_len = min(lengths)
    max_len = max(lengths)
    num_eps = len(trajectories)
    env_name = env.spec.id if hasattr(env, "spec") and env.spec else "unknown"
    date_str = datetime.now().strftime("%Y-%m-%d")

    # Construct filename
    filename = (
        f"traj_{run_id}_date:{date_str}_agent:{agent_type}"
        f"_min:{min_len}_max:{max_len}_episodes:{num_eps}_env:{env_name}.pkl"
    )

    return os.path.join(agent_dir, filename)

def generate_model_filename(base_dir: str,
                            trajectory_id: str,
                            loss_fn_name: str,
                            batch_size: int,
                            optimizer_name: str,
                            embed_dim: int,
                            n_heads: int,
                            n_layers: int,
                            lr: float
                            ) -> str:
    """
    Generate full model file path with metadata-based filename.
    """

    # Ensure directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    # Find next run ID
    run_id = get_next_run_id(base_dir, "models")
    date_str = datetime.now().strftime("%Y-%m-%d")

    filename = (
        f"model_{run_id}_date:{date_str}"
        f"_traj:{trajectory_id}_loss-fn:{loss_fn_name}"
        f"_batch-size:{batch_size}"
        f"_optimizer:{optimizer_name}"
        f"_embed-dim:{embed_dim}"
        f"_n-heads:{n_heads}"
        f"_n-layers:{n_layers}"
        f"_lr:{lr}"
    )

    return os.path.join(base_dir, filename)

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


