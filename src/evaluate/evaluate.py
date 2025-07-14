from evaluate_DT import evaluate_DT
from evaluate_PPO import evaluate_PPO
from decision_transformer import DecisionTransformer
from models import get_models
from datetime import datetime
import utils
import random
import torch
import gymnasium as gym
import os

def evaluate_models(stats_file, date, env: gym.Env, dt_models_names, dt_models, rl_model_names, rl_models, num_episodes: int = 10,
                    max_episode_length: int = 1000):
    os.makedirs(f"results/evaluate/{date}", exist_ok=True)

    stats_file.write("*************************************\n")
    stats_file.write(f"max episode length: {max_episode_length}, episodes: {num_episodes}\n")
    seeds = [random.randrange(0, 2 ** 63 - 1) for _ in range(num_episodes)]
    with open(f"results/evaluate/{date}/DT.txt", "a") as model_file:
        for dt_model_name, dt_model in zip(dt_models_names, dt_models):
            mean_reward_for_step, max_cumulative_reward, min_cumulative_reward, longest_episode =\
                evaluate_DT(model_file, env, dt_model, seeds, num_episodes, max_episode_length)
            stats_file.write(f"{dt_model_name} stats:\n"
                             f"     mean reward for step: {mean_reward_for_step:.3f}\n"
                             f"     max cumulative reward: {max_cumulative_reward:.3f}\n"
                             f"     min cumulative reward: {min_cumulative_reward:.3f}\n"
                     f"     longest episode: {longest_episode:.3f}\n")

    for rl_model_name in rl_model_names:
        rl_model = rl_models[rl_model_name]
        with open(f"results/evaluate/{date}/{rl_model_name}.txt", "a") as model_file:
            mean_reward_for_step, max_cumulative_reward, min_cumulative_reward, longest_episode =\
                evaluate_PPO(model_file, env, rl_model, seeds, num_episodes, max_episode_length)
        stats_file.write(f"{rl_model_name} stats:\n"
                         f"     mean reward for step: {mean_reward_for_step:.3f}\n"
                         f"     max cumulative reward: {max_cumulative_reward:.3f}\n"
                         f"     min cumulative reward: {min_cumulative_reward:.3f}\n"
                         f"     longest episode: {longest_episode:.3f}\n")

    stats_file.write("*************************************\n")


def main():
    # Model parameters
    embed_dim = 128                 # Embedding dimension for the Decision Transformer
    num_layers = 6                  # Number of layers in the Decision Transformer
    num_heads = 8                   # Number of attention heads in the Decision Transformer

    # Trajectory parameters
    num_episodes = 1000
    max_episode_length = 512 

    date = datetime.now().strftime("%d.%m.%Y")
    env = utils.create_environment(env_name='ANM6Easy-v0', entry_point='gym_anm.envs.anm6_env.anm6_easy:ANM6Easy')
    os.makedirs(f"results/evaluate", exist_ok=True)
    os.makedirs(f"results/evaluate/{date}", exist_ok=True)
    
    with open(f"results/evaluate/{date}/evaluate_stats.txt", "a") as evaluate_file:
        evaluate_file.write("Environment created successfully.\n")

        state_dim = 18
        act_dim = 6
        rtg_dim = 1
        boundaries = (env.action_space.low, env.action_space.high)

        dt_versions = [
            "model_6_date:2025-07-12_traj:5_loss-fn:MSELoss_batch-size:128_optimizer:AdamW_embed-dim:128_n-heads:8_n-layers:6_lr:0.0001",
            "val-model_7_date:2025-07-12_traj:5_loss-fn:MSELoss_batch-size:128_optimizer:AdamW_embed-dim:128_n-heads:8_n-layers:6_lr:0.0001"
        ]
        dt_models = []
        dt_models_names = []
        for dt_version in dt_versions:
            dt_model = DecisionTransformer(boundaries, state_dim, act_dim, rtg_dim, embed_dim=embed_dim, n_layer=num_layers, n_head=num_heads, max_episode_len=max_episode_length)
            model_src = f"results/models/DT/{dt_version}"
            dt_model.load_state_dict(torch.load(model_src))
            dt_models.append(dt_model)
            dt_models_names.append('_'.join(dt_version.split('_')[:2]))
            evaluate_file.write(f"DT model {dt_version} loaded successfully.\n")

        rl_model_names = ["PPO", "TD3"]
        rl_models = {}
        models = get_models(env)
        rl_models[rl_model_names[0]] = models[0]
        rl_models[rl_model_names[1]] = models[1] 
        evaluate_file.write("RL models loaded successfully.\n")

        evaluate_models(evaluate_file, date, env, dt_models_names, dt_models, rl_model_names, rl_models, num_episodes=num_episodes, max_episode_length=max_episode_length)

        evaluate_file.write("Evaluation completed successfully.\n")


if __name__ == "__main__":
    main()
