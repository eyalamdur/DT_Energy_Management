from models.ppo import train_ppo
from models.td3 import train_td3
import utils 

def get_models(env) -> list:
    models_dir = "results/models/"

    ppo_model = train_ppo.load_ppo(models_dir + "PPO/ppo_0.zip") if utils.is_model_available(models_dir + "PPO/ppo_0.zip") else train_ppo.train_ppo(env)
    td3_model = train_td3.load_td3(models_dir + "TD3/td3_0.zip") if utils.is_model_available(models_dir + "TD3/td3_0.zip") else train_td3.train_td3(env)
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

    get_models(env)
    utils.color_print("Models loaded or trained successfully.", color='green')

if __name__ == "__main__":
    main()