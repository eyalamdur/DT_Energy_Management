import decision_transformer.utils as utils
import models.ppo.train_ppo as ppo
import models.td3.train_td3 as td3
import utils 


def main():
    """
    Main function to create the environment and train or load the agents.
    This function initializes the environment, checks if the models are available,
    and trains or loads the PPO and TD3 agents accordingly.
    """
    # Create the environment
    env = utils.create_environment(env_name='gym_anm:ANM6Easy-v0', entry_point='gym_anm.envs.anm6_env.anm6_easy:ANM6Easy')
    utils.color_print("Environment created successfully.")

    # train/load the PPO agent
    if (not utils.is_model_available("code/models/ppo/ppo_anm6easy")):
        # Train the PPO agent
        ppo_agent = ppo.train_ppo(env)
    else:
        # Load the PPO agent
        ppo_agent = ppo.load_ppo("code/models/ppo/ppo_anm6easy")
    print("PPO agent is ready.")

    # train/load the TD3 agent
    if (not utils.is_model_available("code/models/td3/td3_anm6easy")):
        # Train the TD3 agent
        td3_agent = td3.train_td3(env)
    else:
        # Load the TD3 agent
        td3_agent = td3.load_td3("code/models/td3/td3_anm6easy")
    print("TD3 agent is ready.")
    
if __name__ == "__main__":
    main()