import gymnasium as gym
from stable_baselines3 import PPO

def train_ppo(env: gym.Env, num_episodes: int = 1000) -> PPO:
    """
    Train a PPO agent on the given environment.
    Args:
        env (gym.Env): The environment to train the agent on.
        num_episodes (int): The number of episodes to train the agent.
    Returns:
        agent (PPO): The trained PPO agent.
    """
    # Create the PPO agent
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=num_episodes)
    model.save("code/models/ppo/ppo_anm6easy")
    
    return model

def load_ppo(model_path: str) -> PPO:
    """
    Load a trained PPO agent from a file.
    Args:
        model_path (str): The path to the saved PPO model.
    Returns:
        agent (PPO): The loaded PPO agent.
    """
    model = PPO.load(model_path)
    return model