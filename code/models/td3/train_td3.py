import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.td3 import MlpPolicy

def train_td3(env: gym.Env, num_episodes: int = 1000) -> TD3:
    """
    Train a TD3 agent on the given environment.
    Args:
        env (gym.Env): The environment to train the agent on.
        num_episodes (int): The number of episodes to train the agent.
    Returns:
        model (TD3): The trained TD3 agent.
    """
    model = TD3(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=num_episodes)
    model.save("code/models/td3/td3_anm6easy")
    return model

def load_td3(model_path: str) -> TD3:
    """
    Load a trained TD3 agent from a file.
    Args:
        model_path (str): The path to the saved TD3 model.
    Returns:
        model (TD3): The loaded TD3 agent.
    """
    model = TD3.load(model_path)
    return model
