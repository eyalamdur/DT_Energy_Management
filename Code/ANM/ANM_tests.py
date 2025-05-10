import gymnasium as gym
import gym_anm
import numpy as np
import torch

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


def get_batch(trajectories: list, batch_size: int = 64, seq_max_len: int = 20, min_traj_length: int = 20):
    """
    Organize the given trajectories into a batches structer.
    Args:
        trajectories (list) : examples of given runs in the environment
        batch_size (int) : the number of sequences (parts of trajectories) in each batch
        seq_max_len (int) : the maximum length of sequences
        min_traj_length (int): minimum size of a trajectory
    Returns:

    """
    # picking only long-enough trajectories
    valid_trajectories = [t for t in trajectories if len(t["states"]) >= min_traj_length]
    if len(valid_trajectories) == 0:
        raise ValueError("No valid trajectories with the required sequence length.")

    # Initialize batches arrays
    state_dim = valid_trajectories[0]['states'].shape[1]
    action_dim = valid_trajectories[0]['actions'].shape[1]

    state_batch = np.zeros((batch_size, seq_max_len, state_dim))
    act_batch = np.zeros((batch_size, seq_max_len, action_dim))
    rtg_batch = np.zeros((batch_size, seq_max_len))
    timestep_batch = np.zeros((batch_size, seq_max_len), dtype=int)
    mask_batch = np.zeros((batch_size, seq_max_len), dtype=int)  # In case of padding

    for i in range(batch_size):
        traj = np.random.choice(valid_trajectories)
        trajectory_len = len(traj["states"])

        # Checking if need to pad the batch with 0-s
        if trajectory_len < seq_max_len:
            si = 0
        else:
            si = np.random.randint(0, trajectory_len - seq_max_len + 1)

        actual_seq_len = min(seq_max_len, trajectory_len - si)

        state_seq = traj["states"][si:si + actual_seq_len]
        act_seq = traj["actions"][si:si + actual_seq_len]
        rtg_seq = traj["rtgs"][si:si + actual_seq_len]
        timestep_seq = np.arange(si, si + actual_seq_len)

        # Fill data
        state_batch[i, :actual_seq_len] = state_seq
        act_batch[i, :actual_seq_len] = act_seq
        rtg_batch[i, :actual_seq_len] = rtg_seq
        timestep_batch[i, :actual_seq_len] = timestep_seq

        # Fill mask: 1 where real, 0 where padded
        mask_batch[i, :actual_seq_len] = 1

    return (
        torch.tensor(state_batch, dtype=torch.float32),
        torch.tensor(act_batch, dtype=torch.long),
        torch.tensor(rtg_batch, dtype=torch.float32),
        torch.tensor(timestep_batch, dtype=torch.long),
        torch.tensor(mask_batch, dtype=torch.long)  # return the mask too
    )

def main():
    # Create the environment
    env = create_environment(env_name='ANM6Easy-v0', entry_point='gym_anm.envs.anm6_env.anm6_easy:ANM6Easy')
    print("Environment created successfully.")

    # Collect trajectories
    trajectories = collect_trajectories(env, num_episodes=20, min_traj_length=1)
    """
    print(f"Collected {len(trajectories)} trajectories.")
    for i, traj in enumerate(trajectories):
        print(f"Trajectory {i + 1}:")
        print(f"States: {traj['states']}")
        print(f"Actions: {traj['actions']}")
        print(f"RTGs: {traj['rtgs']}")
        print()
    """
    # Get a batch
    state_batch, act_batch, rtg_batch, timestep_batch, mask_batch = get_batch(
        trajectories, batch_size=2, seq_max_len=5  # small size for easy printing
    )

    # Print batch shapes
    print("\nBatch shapes:")
    print(f"States batch: {state_batch.shape}")
    print(f"Actions batch: {act_batch.shape}")
    print(f"RTGs batch: {rtg_batch.shape}")
    print(f"Timestep batch: {timestep_batch.shape}")
    print(f"Mask batch: {mask_batch.shape}")

    # Print example batch content
    print("\nFirst batch sample (index 0):")
    print(f"States:\n{state_batch[0]}")
    print(f"Actions:\n{act_batch[0]}")
    print(f"RTGs:\n{rtg_batch[0]}")
    print(f"Timesteps:\n{timestep_batch[0]}")
    print(f"Mask:\n{mask_batch[0]}")

    print("\nSecond batch sample (index 1):")
    print(f"States:\n{state_batch[1]}")
    print(f"Actions:\n{act_batch[1]}")
    print(f"RTGs:\n{rtg_batch[1]}")
    print(f"Timesteps:\n{timestep_batch[1]}")
    print(f"Mask:\n{mask_batch[1]}")

if __name__ == "__main__":
    main()

