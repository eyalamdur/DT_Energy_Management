import gymnasium as gym
from models.anm_exp.rl_agents.train import PPOAgent 

def train_ppo(env :gym.Env, num_episodes: int = 1000) -> PPOAgent:
    """
    Train a PPO agent on the given environment.
    Args:
        env (gym.Env): The environment to train the agent on.
        num_episodes (int): The number of episodes to train the agent.
    Returns:
        agent (PPOAgent): The trained PPO agent.
    """
    # Create the PPO agent
    agent = PPOAgent(env)

    # Training loop
    for episode in range(num_episodes):
        done = False
        obs = env.reset()
        total_reward = 0

        while not done:
            action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            agent.store_transition(obs, action, reward, done)
            total_reward += reward

        agent.train()

        print(f"Episode {episode+1} finished with total reward: {total_reward}")
    
    return agent