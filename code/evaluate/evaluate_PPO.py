import code.models.ppo.train_ppo as ppo
import gymnasium as gym
import numpy as np
import utils

def evaluate_PPO (stats_file, env: gym.Env, model, num_episodes: int = 10, max_episode_length = 1000):
    returns = []

    for episode in range(num_episodes):
        print("episode:", episode)
        stats_file.write(f"episode: {episode}\n")
        ret = episode_evaluation_PPO(stats_file, env, model, max_episode_length)
        returns.append(ret)
        stats_file.write(f"episode {episode} : ret: {ret:.3f}")

    return np.mean(returns)


def episode_evaluation_PPO (stats_file, env: gym.Env, model, max_episode_length = 1000, target_rtg = 100):

    # Initialize evaluate params
    state, _ = env.reset()
    done = False
    cumulative_reword = 0

    # run the test using env.step and sum the rewards
    steps = 0
    while not done and steps < max_episode_length:
        action = model.predict(state)[0]
        state, reward, terminated, truncated, _ = env.step(action)
        utils.print_stats(stats_file, steps, state, action, reward, terminated or truncated)
        cumulative_reword += reward
        steps += 1
        if terminated or truncated:
            done = True

    return cumulative_reword / steps
