import gymnasium as gym
import utils
from tqdm import trange
from models.TrajectoryCallback import TrajectoryCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

MAX_EPISODE_STEPS = 96  # 24 hours × (60 minutes ÷ 15 minutes) = 96 steps per episode


def train_ppo(env: gym.Env, num_episodes: int = 1000) -> (PPO, list):
    """
    Train a PPO agent on the given environment and collect trajectories.
    Args:
        env (gym.Env): The environment to train the agent on.
        num_episodes (int): The number of episodes to train the agent.
    Returns:
        model (PPO): The trained PPO agent.
        trajectories (List[Dict[str, np.ndarray]]): Collected rollouts with keys 'states', 'actions', 'rtgs'.
    """
    # Create PPO agent
    model = PPO(
        policy="MlpPolicy",
        env=env,
        device="cpu",                   # Use CPU for training
        learning_rate=3e-4,             # Use schedule or tune between 1e-4 and 3e-4
        n_steps=4096,                   # Large enough for long-term planning
        batch_size=256,                 # Should divide n_steps evenly
        n_epochs=20,                    # More passes per update for thorough learning
        gamma=0.995,                    # High discount for long-term reward
        gae_lambda=0.97,                # Balanced bias-variance tradeoff
        ent_coef=0.001,                 # Encourage minimal exploration
        verbose=0
    )

    steps_per_episode = env.spec.max_episode_steps or MAX_EPISODE_STEPS
    traj_cb = TrajectoryCallback()

    # Training loop with trajectory collection
    with trange(num_episodes, desc="Training PPO", unit="episode") as pbar:
        for _ in pbar:
            model.learn(
                total_timesteps=steps_per_episode,
                reset_num_timesteps=False,
                callback=traj_cb
            )

    # Save model
    model_id = utils.get_next_run_id("results/models/PPO", "models")
    model.save(f"results/models/PPO/ppo_{model_id}")

    # Return both model and collected trajectories
    return model, traj_cb.trajectories


def load_ppo(model_path: str) -> PPO:
    """
    Load a trained PPO agent from a file.
    Args:
        model_path (str): The path to the saved PPO model.
    Returns:
        agent (PPO): The loaded PPO agent.
    """
    model = PPO.load(model_path, device="cpu")
    return model
