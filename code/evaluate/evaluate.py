from evaluate.evaluate_DT import evaluate_DT
from evaluate.evaluate_PPO import evaluate_PPO
import models.ppo.train_ppo as ppo
import models.td3.train_td3 as td3
from decision_transformer.decision_transformer import DecisionTransformer
import utils
import torch
import gymnasium as gym
import os

def evaluate_models(stats_file, date, env: gym.Env, dt_model_names, dt_models, rl_model_names, rl_models, num_episodes: int = 10,
                    max_episode_length: int = 1000):
    os.makedirs(f"code/evaluate/stats/{date}/DT", exist_ok=True)
    os.makedirs(f"code/evaluate/stats/{date}/RL", exist_ok=True)

    stats_file.write("*************************************\n")
    stats_file.write(f"max episode length: {max_episode_length}, episodes: {num_episodes}\n")

    stats_file.write("EVALUATE DT:\n")
    for dt_model in dt_model_names:
        print(dt_model)
        with open(f"code/evaluate/stats/{date}/DT/{dt_model}.txt", "a") as model_file:
            dt_mean = evaluate_DT(model_file, env, dt_models[dt_model], num_episodes, max_episode_length)
        stats_file.write(f"{dt_model} mean: {dt_mean:.3f}\n")

    stats_file.write("EVALUATE RL:\n")
    for rl_model in rl_model_names:
        print(rl_model)
        with open(f"code/evaluate/stats/{date}/RL/{rl_model}.txt", "a") as model_file:
            rl_mean = evaluate_PPO(model_file, env, rl_models[rl_model], num_episodes, max_episode_length)
        stats_file.write(f"{rl_model} mean: {rl_mean:.3f}\n")

    stats_file.write("*************************************\n")


def main():
    date = "09.06.2025"
    env = utils.create_environment(env_name='ANM6Easy-v0', entry_point='gym_anm.envs.anm6_env.anm6_easy:ANM6Easy')
    os.makedirs(f"code/evaluate/stats", exist_ok=True)
    os.makedirs(f"code/evaluate/stats/{date}", exist_ok=True)
    print("make folders")
    with open(f"code/evaluate/stats/{date}/evaluate_stats.txt", "a") as evaluate_file:
        evaluate_file.write("Environment created successfully.\n")

        state_dim = 18
        act_dim = 6
        rtg_dim = 1
        boundaries = (env.action_space.low, env.action_space.high)

        dt_model_names = ["random", "PPO", "TD3"]
        dt_models = {}
        for model_name in dt_model_names:
            dt_model = DecisionTransformer(boundaries, state_dim, act_dim, rtg_dim)
            model_src = f"logs/dt_models/{model_name}/run_0/model.pt"
            dt_model.load_state_dict(torch.load(model_src))
            dt_models[model_name] = dt_model

        evaluate_file.write("DT models loaded successfully.\n")

        rl_model_names = ["ppo", "td3"]
        rl_models = {}

        if utils.is_model_available("code/models/ppo/ppo_anm6easy"):
            ppo_model = ppo.load_ppo("code/models/ppo/ppo_anm6easy")
        else:
            ppo_model = ppo.train_ppo(env)
        rl_models["ppo"] = ppo_model
        evaluate_file.write("PPO model loaded successfully.\n")
        if utils.is_model_available("code/models/td3/td3_anm6easy"):
            td3_model = td3.load_td3("code/models/td3/td3_anm6easy")
        else:
            td3_model = td3.train_td3(env)
        rl_models["td3"] = td3_model
        evaluate_file.write("TD3 model loaded successfully.\n")

        evaluate_models(evaluate_file, date, env, dt_model_names, dt_models, rl_model_names, rl_models,
                        num_episodes=10, max_episode_length=100)

        evaluate_file.write("Evaluation completed successfully.\n")


if __name__ == "__main__":
    main()
