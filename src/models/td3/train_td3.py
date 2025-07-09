import gymnasium as gym
import utils
from tqdm import trange
from models.TrajectoryCallback import TrajectoryCallback
from stable_baselines3 import TD3
from stable_baselines3.td3 import MlpPolicy

MAX_EPISODE_STEPS = 96  # Maximum steps per episode for TD3 [24 hours × (60 minutes ÷ 15 minutes) = 96 steps per episode]
            
def train_td3(env: gym.Env, num_episodes: int = 20000) -> (TD3, list):
    """
    Train a TD3 agent on the given environment and collect trajectories.
    Args:
        env (gym.Env): The environment to train the agent on.
        num_episodes (int): The number of episodes to train the agent.
    Returns:
        model (TD3): The trained TD3 agent.
        trajectories (List[Dict[str, np.ndarray]]): Collected rollouts with keys 'states', 'actions', 'rtgs'.
    """
        # Create TD3 agent
    model = TD3(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,             # TD3 is less sensitive to LR than PPO
        buffer_size=100_000,            # Larger buffer for richer experience
        batch_size=256,                 # Typical size for stability
        gamma=0.995,                    # Long-term reward focus
        train_freq=(1, "step"),         # Learn once per step (can tune)
        gradient_steps=10,              # Ten updates per step
        learning_starts=10_000,         # Delay learning until buffer is filled
        verbose=1                       # Output for debugging
    )

    steps_per_episode = env.spec.max_episode_steps or MAX_EPISODE_STEPS
    traj_cb = TrajectoryCallback()

    # Training loop with trajectory collection
    with trange(num_episodes, desc="Training TD3", unit="episode") as pbar:
        for _ in pbar:
            model.learn(
                total_timesteps=steps_per_episode,
                reset_num_timesteps=True,
                callback=traj_cb
            )

    # Save model
    model_id = utils.get_next_run_id("results/models/TD3", "models")
    model.save(f"results/models/TD3/td3_{model_id}")
    utils.save_trajectories(traj_cb.trajectories, f"results/trajectories/TD3_train/td3_{model_id}_trajectories.pkl")
    
    return model


def load_td3(model_path: str) -> TD3:
    """
    Load a trained TD3 agent from a file.
    Args:
        model_path (str): The path to the saved TD3 model.
    Returns:
        model (TD3): The loaded TD3 agent.
    """
    model = TD3.load(model_path, device="cpu")
    return model
