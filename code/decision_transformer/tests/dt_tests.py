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
        trajectories[model.__class__.__name__ if model else "random"] = utils.collect_trajectories(env, model=model, num_episodes=10, min_traj_length=20)
        print(f"Collected {len(trajectories[model.__class__.__name__ if model else 'random'])} trajectories for {model.__class__.__name__ if model else 'random'}.")

    # Create the Decision Transformer models
    state_dim = trajectories["random"][0]["states"].shape[1]
    act_dim = trajectories["random"][0]["actions"].shape[1]
    rtg_dim = 1

    random_dt = DecisionTransformer(boundaries, state_dim, act_dim, rtg_dim)
    ppo_dt = DecisionTransformer(boundaries, state_dim, act_dim, rtg_dim)
    td3_dt = DecisionTransformer(boundaries, state_dim, act_dim, rtg_dim)   
    utils.color_print("Models created successfully.")

    # Create the trainers
    utils.color_print("Random DT training:", color="yellow")
    random_trainer = Trainer(random_dt, None, 64)
    random_trainer.train(trajectories["random"])
    utils.color_print("PPO DT training:", color="yellow")
    ppo_trainer = Trainer(ppo_dt, None, 64)
    ppo_trainer.train(trajectories[models[0].__class__.__name__]) 
    utils.color_print("TD3 DT training:", color="yellow")
    td3_trainer = Trainer(td3_dt, None, 64)
    td3_trainer.train(trajectories[models[1].__class__.__name__])

    utils.color_print("Training completed successfully.", color="green")

if __name__ == "__main__":
    main()