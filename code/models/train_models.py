import models.ppo.train_ppo as ppo
import models.td3.train_td3 as td3
import utils 

def get_models(env) -> list:
    ppo_model = ppo.load_ppo("code/models/ppo/ppo_anm6easy") if utils.is_model_available("code/models/ppo/ppo_anm6easy") else ppo.train_ppo(env)
    td3_model = td3.load_td3("code/models/td3/td3_anm6easy") if utils.is_model_available("code/models/td3/td3_anm6easy") else td3.train_td3(env)

    return [ppo_model, td3_model]


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
    ppo_model = ppo.load_ppo("code/models/ppo/ppo_anm6easy") if utils.is_model_available("code/models/ppo/ppo_anm6easy") else ppo.train_ppo(env)
    print("PPO agent is ready.")

    # train/load the TD3 agent
    td3_model = td3.load_td3("code/models/td3/td3_anm6easy") if utils.is_model_available("code/models/td3/td3_anm6easy") else td3.train_td3(env)
    print("TD3 agent is ready.")

if __name__ == "__main__":
    main()