from decision_transformer.decision_transformer import DecisionTransformer
from decision_transformer.trainer import Trainer
from models.train_models import get_models
import utils
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

def generate_trajectories(env):
    trajectories = {}
    models = get_models(env)
    models.append(None)
    for model in models:
        agent_type = model.__class__.__name__ if model else "random"
        traj_data = utils.collect_trajectories(env, model=model, num_episodes=10, min_traj_length=20)
        traj_name = utils.save_trajectories(traj_data, agent_type, env)
        trajectories[agent_type] = (traj_data, traj_name)
    return trajectories


def train_dt_models(trajectories, dt_models):
    """
    Train Decision Transformer models.
    Args:
        trajectories (dict): Dictionary of trajectories for each agent type.
        dt_models (dict): Dictionary of Decision Transformer models.
    """
    # Create the trainers
    for agent_type, model in dt_models.items():
        utils.color_print(f"{agent_type.upper()} DT training:", color="yellow")
        trainer = Trainer(model, None, 64)
        trainer.train(trajectories[agent_type][0], epochs=10)
        print(trajectories[agent_type][1])
        utils.save_model(
            model=model,
            agent_type=agent_type,
            trajectory_path=trajectories[agent_type][1],
            loss_fn_name=trainer.loss_fn.__class__.__name__,
            batch_size=trainer.batch_size,
            optimizer_name=trainer.optimizer.__class__.__name__,
            embed_dim=model.embed_dim,
            n_heads=model.n_head,
            n_layers=model.n_layer,
            lr=trainer.optimizer.param_groups[0]['lr']
        )

def main():
    # Create the environment
    env = utils.create_environment(env_name='ANM6Easy-v0', entry_point='gym_anm.envs.anm6_env.anm6_easy:ANM6Easy')
    utils.color_print("Environment created successfully.")  
    
    # Collect trajectories & get dimensions and The environment's action boundaries
    utils.color_print(f"Collecting trajectories...")
    trajectories = generate_trajectories(env)
    state_dim, act_dim, rtg_dim = get_dimensions(trajectories["random"][0])
    boundaries = env.action_space.low, env.action_space.high 
    
    # Create the Decision Transformer models
    random_dt = DecisionTransformer(boundaries, state_dim, act_dim, rtg_dim)
    ppo_dt = DecisionTransformer(boundaries, state_dim, act_dim, rtg_dim)
    td3_dt = DecisionTransformer(boundaries, state_dim, act_dim, rtg_dim)
    dt_models = {"random": random_dt, "PPO": ppo_dt, "TD3": td3_dt}
    utils.color_print("Models created successfully.")
    
    # Train the Decision Transformer models
    train_dt_models(trajectories, dt_models)
    utils.color_print("Training completed successfully.", color="green")

if __name__ == "__main__":
    main()