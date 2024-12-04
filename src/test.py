from stable_baselines3 import PPO
import os
from environments import TradingEnv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = PPO.load(project_root + '/src/models/ppo_trading', device='cpu')

portfolio_values = []
actions = []
rewards = []

# Load the data
test_df = pd.read_csv(project_root + '/src/data/test.csv')
test_env = TradingEnv(test_df)

obs, info = test_env.reset()
terminated, truncated = False, False
while not (terminated or truncated):
    action = model.predict(obs)[0]
    obs, reward, terminated, truncated, info = test_env.step(action)
    current_step = min(test_env.current_step, len(test_env.data) - 1)
    current_price = test_env.data.iloc[current_step]['Close']
    current_portfolio_value = test_env.cash + test_env.shares * current_price
    portfolio_values.append(current_portfolio_value)
    actions.append(action)
    rewards.append(reward)

# Calculate final portfolio value
final_portfolio_value = portfolio_values[-1]
print(f"Final Portfolio Value: {final_portfolio_value}")

plt.plot(portfolio_values)
plt.title('Portfolio Value Over Time')
plt.xlabel('Time Step')
plt.ylabel('Portfolio Value')
plt.savefig(project_root + '/src/results/portfolio_value.png')
plt.show()

unique, counts = np.unique(actions, return_counts=True)
action_counts = dict(zip(unique, counts))
print(f"Action Counts: {action_counts}")
