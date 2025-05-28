from decision_transformer.decision_transformer import DecisionTransformer
from decision_transformer.trainer import Trainer
import utils
import gym_anm
import torch

def main():
    # Create the environment
    env = utils.create_environment(env_name='ANM6Easy-v0', entry_point='gym_anm.envs.anm6_env.anm6_easy:ANM6Easy')
    print("Environment created successfully.")

    # Collect trajectories
    trajectories = utils.collect_trajectories(env, num_episodes=10, min_traj_length=20)
    print(f"Collected {len(trajectories)} trajectories.")

    # Create the Decision Transformer model
    state_dim = trajectories[0]["states"].shape[1]
    act_dim = trajectories[0]["actions"].shape[1]
    rtg_dim = 1

    # Define the boundaries for actions based on the environment's action space
    boundaries = env.action_space.low, env.action_space.high

    dt_model = DecisionTransformer(boundaries, state_dim, act_dim, rtg_dim)
    print("Model created successfully.")
    # torch.save(dt_model.state_dict(), "random_decision_transformer.pth")

    # Create the trainer
    trainer = Trainer(dt_model, None, 64)
    trainer.train(trajectories)
    print("works:)")

if __name__ == "__main__":
    main()