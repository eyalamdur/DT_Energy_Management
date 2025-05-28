import gymnasium as gym
import numpy as np
import torch
import time

def evaluate_DT (env: gym.Env, model, num_episodes: int = 10, max_episode_length = 1000):
    returns = []

    for episode in range(num_episodes):
        ret = episode_evaluation_DT(env, model, max_episode_length)
        returns.append(ret)

    return np.mean(returns)


def episode_evaluation_DT (env: gym.Env, model, max_episode_length = 1000, target_rtg = 100):

    # Initialize evaluate params
    state, _ = env.reset()
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
        env.render()
        time.sleep(0.5)

        states.append(state)
        actions.append(action)
        rtgs.append(reward if len(rtgs) == 0 else rtgs[-1] + reward)
        timestamp.append(steps)

        if terminated or truncated:
            done = True

        state = next_state
        cumulative_reword += reward
        steps += 1

    return cumulative_reword



