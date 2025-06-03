from evaluate.evaluate_DT import evaluate_DT
from evaluate.evaluate_PPO import evaluate_PPO
import models.ppo.train_ppo as ppo
import models.td3.train_td3 as td3
from decision_transformer.decision_transformer import DecisionTransformer
import decision_transformer.decision_transformer as decision_transformer
import utils
import torch
import gymnasium as gym


def evaluate_models(env: gym.Env, dt_model_names, dt_models, rl_model_names, rl_models, num_episodes: int = 10,
                    max_episode_length: int = 1000):

    print("*************************************")
    print("max episode length: ", max_episode_length, "episodes: ", num_episodes)
    print("EVALUATE DT:")
    for dt_model in dt_model_names:
        print(dt_model, ":")
        dt_mean = evaluate_DT(env, dt_models[dt_model], num_episodes, max_episode_length)
        print(f"{dt_model} mean: {dt_mean:.3f}")
    print("EVALUATE RL:")
    for rl_model in rl_model_names:
        print(rl_model, ":")
        rl_mean = evaluate_PPO(env, rl_models[rl_model], num_episodes, max_episode_length)
        print(f"{rl_model} mean: {rl_mean:.3f}")
    print("*************************************")


def main():

    env = utils.create_environment(env_name='ANM6Easy-v0', entry_point='gym_anm.envs.anm6_env.anm6_easy:ANM6Easy')
    print("Environment created successfully.")

    state_dim = 18
    act_dim = 6
    rtg_dim = 1
    boundaries = env.action_space.low, env.action_space.high

    dt_model_names = ["random", "PPO", "TD3"]
    dt_models = {}
    for model_name in dt_model_names:
        dt_model = DecisionTransformer(boundaries, state_dim, act_dim, rtg_dim)
        model_src = "logs/dt_models/" + model_name + "/run_0/model.pt"
        dt_model.load_state_dict(torch.load(model_src))
        dt_models[model_name] = dt_model

    print("dt models loaded successfully")

    rl_model_names = ["ppo", "td3"]
    rl_models = {}
    ppo_model = ppo.load_ppo("code/models/ppo/ppo_anm6easy") if utils.is_model_available("code/models/ppo/ppo_anm6easy")\
        else ppo.train_ppo(env)
    rl_models["ppo"] = ppo_model
    print("ppo loaded successfully")
    td3_model = td3.load_td3("code/models/td3/td3_anm6easy") if utils.is_model_available("code/models/td3/td3_anm6easy")\
        else ppo.train_ppo(env)
    rl_models["td3"] = td3_model
    print("td3 loaded successfully")
    evaluate_models(env, dt_model_names, dt_models, rl_model_names, rl_models, num_episodes=10, max_episode_length=100)
    print("evaluate successfully")


if __name__ == "__main__":
    main()


