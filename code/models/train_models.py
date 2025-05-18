import argparse
import decision_transformer.utils as utils
import models.ppo.train_ppo as ppo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", default="ppo", help="Agent type to train (default: ppo)")
    args = parser.parse_args()
    
    # Create the environment
    env = utils.create_environment(env_name='ANM6Easy-v0', entry_point='gym_anm.envs.anm6_env.anm6_easy:ANM6Easy')
    print("Environment created successfully.")

    if args.agent == "ppo":
        # train the PPO agent
        ppo_agent = ppo.train_ppo(env)
        print("PPO training completed.")

if __name__ == "__main__":
    main()