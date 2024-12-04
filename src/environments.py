import gymnasium as gym
from gymnasium import spaces
import numpy as np

state_space = ['Close', 'Volume', 'SMA10', 'SMA50', 'Volatility']
action_space = spaces.Discrete(3)

class TradingEnv(gym.Env):
    def __init__(self, data, initial_cash=10000):
        super(TradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.cash = initial_cash
        self.shares = 0
        self.initial_cash = initial_cash
        
        # Define action and observation space
        self.action_space = action_space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(state_space),), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, seed = gym.utils.seeding.np_random(seed)
    
        # Start at a random step within the dataset
        self.current_step = self.np_random.integers(0, len(self.data) - 1)
        self.cash = self.initial_cash
        self.shares = 0
    
        obs = self.data.iloc[self.current_step][state_space].values.astype(np.float32)
        info = {}
        return obs, info


    def step(self, action):
        # Get current price
        current_price = self.data.iloc[self.current_step]['Close']
        noise = self.np_random.normal(0, 0.01 * current_price)  # 1% noise
        current_price += noise
    
        # Validate current_price
        if not np.isfinite(current_price) or current_price <= 0:
            print(f"Invalid current_price at step {self.current_step}: {current_price}")
            current_price = 1e-6
    
        # Record holdings before action
        prev_cash = self.cash
        prev_shares = self.shares
    
        # Compute previous portfolio value using pre-action holdings
        prev_portfolio_value = prev_cash + prev_shares * current_price
    
        # Apply action to update cash and shares
        if action == 1:  # Buy
            shares_to_buy = self.cash // current_price
            self.shares += shares_to_buy
            self.cash -= shares_to_buy * current_price
        elif action == 2:  # Sell
            self.cash += self.shares * current_price
            self.shares = 0
        # action == 0 (Hold) does nothing
    
        # Move to next step
        self.current_step += 1
        terminated = bool(self.current_step >= len(self.data) - 1)
        truncated = False  # Optional: Use if you add max steps logic
    
        # Get next price
        if not terminated:
            next_price = self.data.iloc[self.current_step]['Close']
            noise = self.np_random.normal(0, 0.01 * next_price)  # 1% noise
            next_price += noise
        else:
            next_price = current_price  # Use current_price if at the end
    
        # Validate next_price
        if not np.isfinite(next_price) or next_price <= 0:
            print(f"Invalid next_price at step {self.current_step}: {next_price}")
            next_price = 1e-6
    
        # Compute current portfolio value after action
        portfolio_value = self.cash + self.shares * next_price
    
        # Calculate reward
        reward = portfolio_value - prev_portfolio_value
    
        # Clip reward to prevent instability
        reward = np.clip(reward, -1e6, 1e6)
    
        # Get next state
        if not terminated:
            next_state = self.data.iloc[self.current_step][state_space].values.astype(np.float32)
        else:
            next_state = self.data.iloc[self.current_step - 1][state_space].values.astype(np.float32)
    
        # Validate next_state
        if not np.all(np.isfinite(next_state)):
            print(f"Invalid next_state at step {self.current_step}: {next_state}")
            next_state = np.zeros_like(next_state)  # Fallback to zero state
    
        info = {}
        return next_state, reward, terminated, truncated, info
