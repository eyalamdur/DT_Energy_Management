import numpy as np
import gymnasium as gym

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

def collect_trajectories(env: gym.Env, num_episodes: int = 500, min_traj_length: int = 20) -> list :
    """
    Collect trajectories from the specified environment.
    Args:
        env (str): The name of the environment to collect trajectories from.
        num_episodes (int): The number of episodes to collect.
        min_traj_length (int): minimum size of a trajectory
    Returns:
        trajectories (list): A list of collected trajectories.
    """
    # Collect trajectories
    trajectories = []
    for _ in range(num_episodes):
        print(f"Collecting episode {_ + 1}/{num_episodes}")
        obs, _ = env.reset()
        done = False
        states, actions, rewards = [], [], []
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            obs = next_obs
        # filter out short episodes
        if len(states) >= min_traj_length:
            rtgs = np.cumsum(rewards[::-1])[::-1]
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
    
