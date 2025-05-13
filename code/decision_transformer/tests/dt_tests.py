import os
import sys
# Add the "code" folder to the Python path TODO FIX THIS
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from decision_transformer.decision_transformer import DecisionTransformer
from decision_transformer.trainer import Trainer
import gymnasium as gym
import decision_transformer.utils as utils
import gym_anm

def create_environment(env_name : str, entry_point : str) -> gym.Env:
    """
    Create a gym environment with the specified name and maximum episode steps.
    Args:
        env_name (str): The name of the environment to create.
        entry_point (str): The entry point for the environment.
    Returns:
        env (gym.Env): The created gym environment.
    """
    
    # Register the environment if it is not already registered
    if env_name not in gym.envs.registry:
        print(f"Registering environment '{env_name}'.")
        gym.register(id=env_name,entry_point=entry_point)
        
    # Create and return the environment
    env = gym.make(env_name)
    return env

def main():
    # Create the environment
    env = create_environment(env_name='ANM6Easy-v0', entry_point='gym_anm.envs.anm6_env.anm6_easy:ANM6Easy')
    print("Environment created successfully.")

    # Collect trajectories
    trajectories = utils.collect_trajectories(env, num_episodes=10, min_traj_length=20)
    print(f"Collected {len(trajectories)} trajectories.")

    # Create the Decision Transformer model
    state_dim = trajectories[0]["states"].shape[1]
    act_dim = trajectories[0]["actions"].shape[1]
    rtg_dim = 1

    dt_model = DecisionTransformer(state_dim, act_dim, rtg_dim)
    print ("Model created successfully.")
    
    # Create the trainer
    trainer = Trainer(dt_model, None, 64)
    trainer.train(trajectories)
    print("works:)")

if __name__ == "__main__":
    main()