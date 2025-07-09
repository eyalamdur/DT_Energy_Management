import models.ppo.train_ppo as ppo
import gymnasium as gym
import numpy as np
import utils

def evaluate_PPO (stats_file, env: gym.Env, model, seeds, num_episodes: int = 10, max_episode_length = 1000):
    cumulative_rewords, mean_rewards, episodes_lengths = [], [], []

    for episode in range(num_episodes):
        print("episode:", episode)
        stats_file.write(f"episode: {episode} seed: {seeds[episode]}\n")
        cumulative_reword, mean_reward, episode_length = episode_evaluation_PPO(stats_file, env, model,
                                                                                  seeds[episode], max_episode_length)
        cumulative_rewords.append(cumulative_reword)
        mean_rewards.append(mean_reward)
        episodes_lengths.append(episode_length)
        stats_file.write(f"episode {episode} : cumulative_reword: {cumulative_reword:.3f}, mean_reward: {mean_reward:.3f}, steps: {episode_length}")

    return np.mean(mean_rewards), np.max(cumulative_rewords), np.min(cumulative_rewords), np.max(episodes_lengths)


def episode_evaluation_PPO (stats_file, env: gym.Env, model, seed, max_episode_length = 1000, target_rtg = 100):

    # Initialize evaluate params
    state, _ = env.reset(seed=seed)
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

    return cumulative_reword, cumulative_reword / steps, steps
