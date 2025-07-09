import gymnasium as gym
import numpy as np
import torch
import utils


def evaluate_DT (stats_file, env: gym.Env, model, seeds, num_episodes: int = 10, max_episode_length = 1000):
    cumulative_rewords, mean_rewards, episodes_lengths = [], [], []

    for episode in range(num_episodes):
        print("episode:", episode)
        stats_file.write(f"episode: {episode}, seed: {seeds[episode]}\n")
        cumulative_reword, mean_reward, episode_length = episode_evaluation_DT(stats_file, env, model,
                                                                                seeds[episode], max_episode_length)
        cumulative_rewords.append(cumulative_reword)
        mean_rewards.append(mean_reward)
        episodes_lengths.append(episode_length)
        stats_file.write(f"episode {episode} : cumulative_reword: {cumulative_reword:.3f}, mean_reward: {mean_reward:.3f}, steps: {episode_length}")

    return np.mean(mean_rewards), np.max(cumulative_rewords), np.min(cumulative_rewords), np.max(episodes_lengths)


def episode_evaluation_DT (stats_file, env: gym.Env, model, seed, max_episode_length = 1000, target_rtg = 1000):

    # Initialize evaluate params
    state, _ = env.reset(seed=seed)
    done = False
    cumulative_reword = 0
    states, actions, rtgs, timestamp = [], [], [], []
    # run the test using env.step and sum the rewards
    steps = 0
    while not done and steps < max_episode_length:
        # takes the last sequence (state, action, reward) into tensors,
        # if it's the first step - pad the action and reword

        state_tensor = torch.tensor(state, dtype=torch.float32).reshape(1, 1, -1)
        if len(actions) == 0:
            action_tensor = torch.zeros((1, 1, model.act_dim), dtype=torch.float32)
        else:
            action_tensor = torch.tensor(actions[-1], dtype=torch.float32).reshape(1, 1, -1)

        if len(rtgs) == 0:
            rtg_tensor = torch.tensor([[target_rtg]], dtype=torch.float32)
        else:
            rtg_tensor = torch.tensor([[rtgs[-1]]], dtype=torch.float32)

        timestep_tensor = torch.tensor([[steps]], dtype=torch.long)

        # getting the action prediction from the model
        action = model.get_action(state_tensor, action_tensor, rtg_tensor, timestep_tensor)
        next_state, reward, terminated, truncated, _ = env.step(action)
        utils.print_stats(stats_file, steps, state, action, reward, terminated or truncated)
        states.append(state)
        actions.append(action)
        rtgs.append(reward if len(rtgs) == 0 else rtgs[-1] + reward)
        timestamp.append(steps)

        if terminated or truncated:
            done = True

        state = next_state
        cumulative_reword += reward
        steps += 1

    return cumulative_reword, cumulative_reword / steps, steps

