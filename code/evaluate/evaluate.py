from evaluate.evaluate_DT import evaluate_DT
from evaluate.evaluate_PPO import evaluate_PPO
import models.ppo.train_ppo as ppo
from decision_transformer.decision_transformer import DecisionTransformer
import decision_transformer.decision_transformer as decision_transformer
import utils
import torch
import gymnasium as gym


def evaluate_models(env: gym.Env, dt_model, ppo_model, num_episodes: int = 10, max_episode_length = 1000):

    dt_mean = evaluate_DT(env, dt_model, num_episodes, max_episode_length)
    ppo_mean = evaluate_PPO(env, ppo_model, num_episodes, max_episode_length)

    print("*************************************")
    print("EVALUATE DT VS. PPO:")
    print("max episode length: ", max_episode_length, "episodes: ", num_episodes)
    print("dt score: ", dt_mean, " ppo score: ", ppo_mean)
    print("*************************************")

def main():

    env = utils.create_environment(env_name='ANM6Easy-v0', entry_point='gym_anm.envs.anm6_env.anm6_easy:ANM6Easy')
    print("Environment created successfully.")

    state_dim = 18
    act_dim = 6
    rtg_dim = 1
    boundaries = env.action_space.low, env.action_space.high

    dt_model = DecisionTransformer(boundaries, state_dim, act_dim, rtg_dim)
    dt_model.load_state_dict(torch.load("code/decision_transformer.pth"))
    print("dt loaded successfully")

    ppo_model = ppo.load_ppo("code/models/ppo/ppo_anm6easy") if utils.is_model_available("code/models/ppo/ppo_anm6easy")\
        else ppo.train_ppo(env)
    print("ppo loaded successfully")

    evaluate_models(env, dt_model, ppo_model, num_episodes=5, max_episode_length=50)
    print("evaluate successfully")

if __name__ == "__main__":
    main()


