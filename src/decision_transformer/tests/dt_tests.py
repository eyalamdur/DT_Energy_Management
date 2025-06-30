from decision_transformer import DecisionTransformer, Trainer
from models.train_models import get_models
import utils
import torch
import gym_anm
import torch

def get_dimensions(trajectories : list) -> tuple:
    """
    Get the dimensions of states, actions, and rewards from the trajectories.
    Args:
        trajectories (List[Dict[str, np.ndarray]]): List of trajectories.
    Returns:
        Tuple[int, int, int]: Dimensions of states, actions, and rewards.
    """
    state_dim = trajectories[0]["states"].shape[1]
    act_dim = trajectories[0]["actions"].shape[1]
    rtg_dim = 1  # Assuming return-to-go is a scalar
    return state_dim, act_dim, rtg_dim

def generate_trajectories(env, num_episodes, min_traj_length, max_traj_length):
    """
    Generate trajectories for different agent types in the environment.
    Args:
        env (gym.Env): The environment to collect trajectories from.
        num_episodes (int): Number of episodes to collect for each agent type.
        min_traj_length (int): Minimum trajectory length.
        max_traj_length (int): Maximum trajectory length.
    Returns:
        List[Dict[str, np.ndarray]]: List of trajectories of all the agents concatenated.
        str: The ID of the saved trajectories.
    """
    trajectories = []
    models = get_models(env)
    models.append(None)
    for model in models:
        agent_type = model.__class__.__name__ if model else "random"
        traj_data = utils.collect_trajectories(env, model=model, num_episodes=num_episodes, min_traj_length=min_traj_length, max_traj_length=max_traj_length)
        traj_id = int(utils.save_trajectories(traj_data, agent_type, env).split("_")[1]) if traj_data else None
        trajectories += traj_data
        
    return (trajectories, traj_id)

def load_trajectories(env, traj_id):
    """
    Load trajectories for ppo, random, and td3 agent types using the given traj_id.
    Args:
        env (gym.Env): The environment instance (for path resolution).
        traj_id (int or str): The trajectory ID to load.
    Returns:
        List[Dict[str, np.ndarray]]: List of trajectories of all the agents concatenated.
        str: The ID of the loaded trajectories.
    """
    agent_types = ["PPO", "random", "TD3"]
    all_trajectories = []
    for agent_type in agent_types:
        traj_data = utils.load_trajectories(agent_type, traj_id)
        if traj_data:
            all_trajectories += traj_data
    return (all_trajectories, traj_id)

def train_dt_model(trajectories, dt_model, batch_size, loss_fn, epochs, device, min_traj_length, max_traj_length):
    """
    Train Decision Transformer models.
    Args:
        trajectories (dict): Dictionary of trajectories for each agent type.
        dt_models (dict): Dictionary of Decision Transformer models.
    """
    # Create the trainer
    trainer = Trainer(dt_model, None, batch_size, loss_fn=loss_fn, device=device)
    trainer.train(trajectories[0], epochs=epochs, min_traj_length=min_traj_length, max_traj_length=max_traj_length)
    utils.save_model(
        model=dt_model,
        trajectory_id=trajectories[1],
        loss_fn_name=trainer.loss_fn.__class__.__name__,
        batch_size=trainer.batch_size,
        optimizer_name=trainer.optimizer.__class__.__name__,
        embed_dim=dt_model.embed_dim,
        n_heads=dt_model.n_head,
        n_layers=dt_model.n_layer,
        lr=trainer.optimizer.param_groups[0]['lr']
        )

def main():
    
    # Trajectory parameters
    num_episodes = 2000             # Number of episodes to collect for each agent type
    load = True                     # Load existing trajectories if available
    traj_id = 3                     # ID of the trajectory to load or save
    min_traj_length = 96            # 1 day in ANM6Easy-v0
    max_traj_length = 480           # 5 days in ANM6Easy-v0
    
    # Model parameters
    training_epochs = 10000
    batch_size = 256                # Batch size for training
    embed_dim = 256                 # Embedding dimension for the Decision Transformer
    num_layers = 6                  # Number of layers in the Decision Transformer
    num_heads = 8                   # Number of attention heads in the Decision Transformer
    loss_fn = torch.nn.MSELoss()
    
    # Set the device for PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the environment
    env = utils.create_environment(env_name='ANM6Easy-v0', entry_point='gym_anm.envs.anm6_env.anm6_easy:ANM6Easy')
    utils.color_print("Environment created successfully.")  
    
    # Collect trajectories & get dimensions and The environment's action boundaries
    utils.color_print(f"Collecting trajectories...")
    trajectories = load_trajectories(env, traj_id=traj_id) if load else generate_trajectories(env, num_episodes, min_traj_length, max_traj_length)
    state_dim, act_dim, rtg_dim = get_dimensions(trajectories[0])
    boundaries = env.action_space.low, env.action_space.high 
    
    # Create the Decision Transformer models
    dt = DecisionTransformer(boundaries, state_dim, act_dim, rtg_dim, embed_dim=embed_dim, n_layer=num_layers, n_head=num_heads, max_episode_len=max_traj_length).to(device)
    dt.transformer.gradient_checkpointing_enable()      # Enable gradient checkpointing for memory efficiency
    utils.color_print("Models created successfully.")
    
    # Train the Decision Transformer models
    train_dt_model(trajectories, dt, batch_size=batch_size, loss_fn=loss_fn, epochs=training_epochs, device=device, min_traj_length=min_traj_length, max_traj_length=max_traj_length)
    utils.color_print("Training completed successfully.", color="green")

if __name__ == "__main__":
    main()