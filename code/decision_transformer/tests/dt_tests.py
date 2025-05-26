from decision_transformer.decision_transformer import DecisionTransformer
from decision_transformer.trainer import Trainer
import decision_transformer.utils as utils
import gym_anm

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

    dt_model = DecisionTransformer(state_dim, act_dim, rtg_dim)
    print ("Model created successfully.")
    
    # Create the trainer
    trainer = Trainer(dt_model, None, 64)
    trainer.train(trajectories)
    print("works:)")

if __name__ == "__main__":
    main()