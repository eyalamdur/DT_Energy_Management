import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import time

# ---------- Collect Trajectories ----------
def collect_trajectories(env_name, num_episodes=100):
    env = gym.make(env_name)
    trajectories = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        states, actions, rewards = [], [], []

        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(obs)
            actions.append(action)
            rewards.append(reward)

            obs = next_obs

        if len(states) >= 20:  # filter out short episodes
            rtgs = np.cumsum(rewards[::-1])[::-1]
            trajectories.append({
                "states": np.array(states),
                "actions": np.array(actions),
                "rtgs": rtgs
            })

    return trajectories


# ---------- Batch Prepping ----------
def get_batch(trajectories, batch_size=64, seq_len=20):
    valid_trajectories = [t for t in trajectories if len(t["states"]) >= seq_len]
    if len(valid_trajectories) == 0:
        raise ValueError("No valid trajectories with the required sequence length.")

    state_dim = valid_trajectories[0]['states'].shape[1]
    obs_batch = np.zeros((batch_size, seq_len, state_dim))
    act_batch = np.zeros((batch_size, seq_len), dtype=int)
    rtg_batch = np.zeros((batch_size, seq_len))
    timestep_batch = np.zeros((batch_size, seq_len), dtype=int)

    for i in range(batch_size):
        traj = np.random.choice(valid_trajectories)

        # Ensure there are enough states to slice
        trajectory_len = len(traj["states"])
        if trajectory_len <= seq_len:
            si = 0
        else:
            si = np.random.randint(0, trajectory_len - seq_len)

        obs_batch[i] = traj["states"][si:si + seq_len]
        act_batch[i] = traj["actions"][si:si + seq_len]
        rtg_batch[i] = traj["rtgs"][si:si + seq_len]
        timestep_batch[i] = np.arange(si, si + seq_len)

    return (
        torch.tensor(obs_batch, dtype=torch.float32),
        torch.tensor(act_batch, dtype=torch.long),
        torch.tensor(rtg_batch, dtype=torch.float32),
        torch.tensor(timestep_batch, dtype=torch.long)
    )

# ---------- Model ----------
class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, embed_dim=128, seq_len=20):
        super().__init__()
        self.state_embed = nn.Linear(state_dim, embed_dim)
        self.action_embed = nn.Embedding(act_dim, embed_dim)
        self.rtg_embed = nn.Linear(1, embed_dim)
        self.timestep_embed = nn.Embedding(1024, embed_dim)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4),
            num_layers=3
        )

        self.predict_action = nn.Linear(embed_dim, act_dim)

    def forward(self, states, actions, rtgs, timesteps):
        batch_size, seq_len = states.shape[0], states.shape[1]

        state_embeddings = self.state_embed(states)
        action_embeddings = self.action_embed(actions)
        rtg_embeddings = self.rtg_embed(rtgs.unsqueeze(-1))
        timestep_embeddings = self.timestep_embed(timesteps)

        tokens = state_embeddings + action_embeddings + rtg_embeddings + timestep_embeddings
        tokens = tokens.transpose(0, 1)  # [seq_len, batch, embed]

        x = self.transformer(tokens)  # [seq_len, batch, embed]
        x = x.transpose(0, 1)

        return self.predict_action(x)

# ---------- Training ----------
def train(model, trajectories, epochs=1000, batch_size=64, seq_len=20, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        states, actions, rtgs, timesteps = get_batch(trajectories, batch_size, seq_len)
        preds = model(states, actions, rtgs, timesteps)

        loss = loss_fn(preds.view(-1, preds.shape[-1]), actions.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ---------- Testing ----------
def test_decision_transformer(env, model, target_return, K=20):
    obs, _ = env.reset()
    obs = obs.astype(np.float32)

    states, actions, rtgs, timesteps = [], [], [target_return], []

    total_reward = 0
    done = False

    for t in range(1000):
        # Update history
        states.append(obs)
        ts = t
        timesteps.append(ts)

        # Pad or trim to K
        def pad(arr, shape=None, pad_value=0):
            arr = arr[-K:]
            if len(arr) < K:
                pad_len = K - len(arr)
                if shape:
                    arr = [np.zeros(shape, dtype=np.float32)] * pad_len + arr
                else:
                    arr = [pad_value] * pad_len + arr
            return arr

        s_np = pad(states, shape=obs.shape)
        a_np = pad(actions, pad_value=0)
        r_np = pad(rtgs, pad_value=target_return)
        ts_np = pad(timesteps, pad_value=0)

        s = torch.tensor(np.array(s_np), dtype=torch.float32).unsqueeze(0)  # [1, K, state_dim]
        a = torch.tensor(a_np, dtype=torch.long).unsqueeze(0)               # [1, K]
        r = torch.tensor(r_np, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # [1, K, 1]
        ts = torch.tensor(ts_np, dtype=torch.long).unsqueeze(0)            # [1, K]

        # print(f"State shape: {s.shape}")
        # print(f"Action shape: {a.shape}")
        # print(f"Return shape: {r.shape}")
        # print(f"Timestep shape: {ts.shape}")

        with torch.no_grad():
            pred = model(s, a, r.squeeze(-1), ts)
            act = torch.argmax(pred[0, -1]).item()

        next_obs, reward, terminated, truncated, _ = env.step(act)
        done = terminated or truncated

        total_reward += reward
        actions.append(act)
        rtgs.append(rtgs[-1] - reward)
        obs = next_obs.astype(np.float32)

        if done:
            break
    time.sleep(5)
    return total_reward

# ---------- MAIN ----------
def main():
    env_name = "CartPole-v1"
    K = 20
    print("Collecting trajectories...")
    trajectories = collect_trajectories(env_name, num_episodes=100)

    state_dim = trajectories[0]["states"].shape[1]
    act_dim = int(np.max([traj["actions"].max() for traj in trajectories])) + 1

    model = DecisionTransformer(state_dim, act_dim, seq_len=K)
    print("Training model...")
    train(model, trajectories, epochs=1000, seq_len=K)

    print("Saving model...")
    torch.save(model.state_dict(), "dt_cartpole.pt")

    print("Loading model and testing...")
    model.load_state_dict(torch.load("dt_cartpole.pt"))
    env = gym.make(env_name, render_mode="human")

    reward = test_decision_transformer(env, model, target_return=30, K=K)
    print(f"Test episode reward: {reward}")

if __name__ == "__main__":
    main()
