from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from environments import TradingEnv
import pandas as pd
import os
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(project_root + '/config.yaml') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Load the data
train_df = pd.read_csv(project_root + '/src/data/train.csv')

# Create the environment
env = TradingEnv(train_df)

# Ensure the environment adheres to Gym API standards
check_env(env, warn=True)

# Create the model
model = PPO("MlpPolicy", env, verbose=1, device='cpu')

# Train the agent
model.learn(total_timesteps=config['hyperparameters']['total_timesteps'])

model_save_path = project_root + '/src/models/ppo_trading'
model.save(model_save_path)

print(f"Training Complete! Model saved to {model_save_path}")
