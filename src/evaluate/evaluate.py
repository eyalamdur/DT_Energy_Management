from evaluate_DT import evaluate_DT
from evaluate_PPO import evaluate_PPO
from decision_transformer import DecisionTransformer
from models import get_models
from datetime import datetime
import utils
import torch
import gymnasium as gym
import os

def evaluate_models(stats_file, date, env: gym.Env, dt_model, rl_model_names, rl_models, num_episodes: int = 10,
                    max_episode_length: int = 1000):
    os.makedirs(f"src/evaluate/stats/{date}", exist_ok=True)

    stats_file.write("*************************************\n")
    stats_file.write(f"max episode length: {max_episode_length}, episodes: {num_episodes}\n")

    stats_file.write("EVALUATE DT:\n")
    with open(f"src/evaluate/stats/{date}/DT.txt", "a") as model_file:
        dt_mean = evaluate_DT(model_file, env, dt_model, num_episodes, max_episode_length)
    stats_file.write(f"DT mean: {dt_mean:.3f}\n")

    stats_file.write("EVALUATE RL:\n")
    for rl_model in rl_model_names:
        print(rl_model)
        with open(f"src/evaluate/stats/{date}/{rl_model}.txt", "a") as model_file:
            rl_mean = evaluate_PPO(model_file, env, rl_models[rl_model], num_episodes, max_episode_length)
        stats_file.write(f"{rl_model} mean: {rl_mean:.3f}\n")

    stats_file.write("*************************************\n")


def main():
    # Model parameters
    embed_dim = 256                 # Embedding dimension for the Decision Transformer
    num_layers = 6                  # Number of layers in the Decision Transformer
    num_heads = 8                   # Number of attention heads in the Decision Transformer

    # Trajectory parameters
    num_episodes = 1000 #10
    max_episode_length = 480 #480

    date = datetime.now().strftime("%d.%m.%Y")
    env = utils.create_environment(env_name='ANM6Easy-v0', entry_point='gym_anm.envs.anm6_env.anm6_easy:ANM6Easy')
    os.makedirs(f"src/evaluate/stats", exist_ok=True)
    os.makedirs(f"src/evaluate/stats/{date}", exist_ok=True)
    
    with open(f"src/evaluate/stats/{date}/evaluate_stats.txt", "a") as evaluate_file:
        evaluate_file.write("Environment created successfully.\n")

        state_dim = 18
        act_dim = 6
        rtg_dim = 1
        boundaries = (env.action_space.low, env.action_space.high)

        dt_version = "model_3_date:2025-06-28_traj:3_loss-fn:MSELoss_batch-size:256_optimizer:AdamW_embed-dim:256_n-heads:8_n-layers:6_lr:0.0001"
        # dt_model = DecisionTransformer(boundaries, state_dim, act_dim, rtg_dim)
        dt_model = DecisionTransformer(boundaries, state_dim, act_dim, rtg_dim, embed_dim=embed_dim, n_layer=num_layers, n_head=num_heads, max_episode_len=480)
        model_src = f"results/models/DT/{dt_version}"
        dt_model.load_state_dict(torch.load(model_src))
        evaluate_file.write("DT models loaded successfully.\n")

        rl_model_names = ["PPO", "TD3"]
        rl_models = {}
        models = get_models(env)
        rl_models[rl_model_names[0]] = models[0]
        rl_models[rl_model_names[1]] = models[1] 
        evaluate_file.write("RL models loaded successfully.\n")

        evaluate_models(evaluate_file, date, env, dt_model, rl_model_names, rl_models, num_episodes=num_episodes, max_episode_length=max_episode_length)

        evaluate_file.write("Evaluation completed successfully.\n")


if __name__ == "__main__":
    main()
