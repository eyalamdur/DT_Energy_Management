from decision_transformer.decision_transformer import DecisionTransformer
from decision_transformer.trainer import Trainer
from models.train_models import get_models
import utils
import gym_anm

def main():
    # Create the environment
    env = utils.create_environment(env_name='ANM6Easy-v0', entry_point='gym_anm.envs.anm6_env.anm6_easy:ANM6Easy')
    utils.color_print("Environment created successfully.")

    # Define the boundaries for actions based on the environment's action space
    boundaries = env.action_space.low, env.action_space.high
    
    # Collect trajectories
    utils.color_print(f"Collecting trajectories...")
    trajectories = {}
    models = get_models(env)
    models.append(None)
    for model in models:
        agent_type = model.__class__.__name__ if model else "random"
        traj_data = utils.collect_trajectories(env, model=model, num_episodes=10, min_traj_length=20)
        utils.save_trajectories(traj_data, agent_type)
        trajectories[agent_type] = traj_data

    # Create the Decision Transformer models
    state_dim = trajectories["random"][0]["states"].shape[1]
    act_dim = trajectories["random"][0]["actions"].shape[1]
    rtg_dim = 1

    random_dt = DecisionTransformer(boundaries, state_dim, act_dim, rtg_dim)
    ppo_dt = DecisionTransformer(boundaries, state_dim, act_dim, rtg_dim)
    td3_dt = DecisionTransformer(boundaries, state_dim, act_dim, rtg_dim)   
    utils.color_print("Models created successfully.")

    # Create the trainers
    dt_models = {"random": random_dt,"PPO": ppo_dt,"TD3": td3_dt}
    
    for agent_type, model in dt_models.items():
        utils.color_print(f"{agent_type.upper()} DT training:", color="yellow")
        trainer = Trainer(model, None, 64)
        trainer.train(trajectories[agent_type])
        utils.save_model(model, agent_type)

    utils.color_print("Training completed successfully.", color="green")

if __name__ == "__main__":
    main()