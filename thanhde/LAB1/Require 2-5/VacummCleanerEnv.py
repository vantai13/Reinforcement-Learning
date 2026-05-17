# Build a simple custom Gymnasium environment named "VacuumCleaner-v0".
# The environment simulates a vacuum robot operating in an m x n room. The robot can
# move up, down, left, and right and automatically vacuums the cell it occupies.
# The objective is to clean all dust particles in the room. There is a single obstacle
# located at a specified cell (i, j) that the robot must avoid. Entering the obstacle
# cell yields a large negative reward and terminates the episode.
# The robot receives a positive reward when it vacuums a dirty cell. If the robot
# attempts to vacuum an already clean cell, that action receives a reduced reward
# (e.g., penalized or halved). When all dust has been cleaned, the agent receives
# a large positive bonus reward and the episode terminates.
# Action space: Discrete(4) -> {0: up, 1: down, 2: left, 3: right}
# Observation space: Dict with 'position' (x, y) and 'dust' grid (m x n binary)

import gymnasium as gym
import numpy as np
import os
import time
from IPython.display import clear_output

class VacuumCleanerEnv(gym.Env):
    def __init__(self, m=5, n=5, obstacle=(2, 2)):
        super(VacuumCleanerEnv, self).__init__()
        self.m = m
        self.n = n
        self.obstacle = tuple(obstacle)

        # Action space: 0=up, 1=down, 2=left, 3=right
        ### YOU NEED TO WRITE YOUR CODE BELOW ###
        # HERE
        # Thiết lập không gian action cho agent là 4 (0, 1, 2, 3)
        self.action_space = gym.spaces.Discrete(4)

        # Observation space: position and dust grid
        ### YOU NEED TO WRITE YOUR CODE BELOW ###
        # HERE
        # Thiết lập không gian giám sát của env là vị trí của robot và ma trận bụi
        self.observation_space = gym.spaces.Dict({
            'position': gym.spaces.Box(low=0, high=max(m-1,n-1), shape=(2,)),
            'dust': gym.spaces.Box(low=0, high=1, shape=(m,n))
        })

        self.reset()

    def reset(self, *, seed=None, options=None):
        # initialize position and dust
        # Start the robot at the top-left corner (row 0, column 0)
        # Use NUMPY to define.
        ### YOU NEED TO WRITE YOUR CODE BELOW ###
        # HERE
        # Tạo ra mảng có 2 phần tử bằng thư viện Numpy
        self.position = np.array([0,0])

        # Initialize dust grid: 1 indicates dirty, 0 indicates clean.
        # Shape is (m, n) corresponding to the room dimensions.
        # Use NUMPY to define.
        ### YOU NEED TO WRITE YOUR CODE BELOW ###
        # HERE
        # Tạo ra ma trận mxn với tất cả các phần tử đều bằng 1 bằng thư viện Numpy
        self.dust_grid = np.ones((self.m, self.n))

        # Ensure the obstacle cell contains no dust (robot cannot clean there).
        # This also prevents rewarding the agent for occupying the obstacle.
        self.dust_grid[self.obstacle] = 0  # obstacle cell has no dust
        self.total_reward = 0.0
        self.truncated = False
        self.terminated = False
        obs = {'position': self.position.copy(), 'dust': self.dust_grid.copy()}
        return obs, {}

    def step(self, action):
        # compute candidate new position
        ### YOU NEED TO WRITE YOUR CODE BELOW ###
        if action == 0:   # Up
            candidate = [self.position[0] - 1, self.position[1]]
        elif action == 1: # Down
            candidate = [self.position[0] + 1, self.position[1]]
        elif action == 2: # Left
            candidate = [self.position[0], self.position[1] - 1]
        elif action == 3: # Right
            candidate = [self.position[0], self.position[1] + 1]
        else:
            candidate = self.position.copy()

        # boundary check
        if (0 <= candidate[0] < self.m) and (0 <= candidate[1] < self.n):
            # obstacle check
            if tuple(candidate) == self.obstacle:
                self.position = candidate.copy()
                reward = -10.0
                self.terminated = True
                obs = {'position': self.position.copy(), 'dust': self.dust_grid.copy()}
                self.total_reward += reward
                return obs, reward, True, False, {}
            else:
                self.position = candidate.copy()
        # else: stay in place

        # If the robot is on a dirty cell, give a positive reward (1.0) and mark it clean.
        # If the cell is already clean, apply a small penalty (-0.5) to discourage redundant cleaning.
        ### YOU NEED TO WRITE YOUR CODE BELOW ###
        # HERE
        if self.dust_grid[self.position[0], self.position[1]] == 1:
            reward = 1
            self.dust_grid[self.position[0], self.position[1]] = 0
        else: reward = -0.5

        self.total_reward += reward

        # check if all cleaned
        if np.sum(self.dust_grid) == 0:
            reward += 10.0
            self.total_reward += 10
            self.terminated = True

        obs = {'position': self.position.copy(), 'dust': self.dust_grid.copy()}
        return obs, reward, bool(self.terminated), bool(self.truncated), {}

    def render(self, mode='human'):
        # In Jupyter notebooks, use IPython.display.clear_output to clear the cell output.
        try:
            clear_output(wait=True)
        except Exception:
            # Fallback for terminal execution
            os.system('cls' if os.name == 'nt' else 'clear')

        # Build display grid with symbols:
        # '#' obstacle, '.' dirty, ' ' clean, 'R' robot, 'X' robot on obstacle
        display = np.full((self.m, self.n), ' ', dtype='<U1')
        for i in range(self.m):
            for j in range(self.n):
                if (i, j) == self.obstacle:
                    display[i, j] = '#'
                elif self.dust_grid[i, j] == 1:
                    display[i, j] = '.'
                else:
                    display[i, j] = ' '

        x, y = int(self.position[0]), int(self.position[1])
        if (x, y) == self.obstacle:
            display[x, y] = 'X'
        else:
            display[x, y] = 'R'

        for row in display:
            print(''.join(row))
        print(f"Total reward: {self.total_reward}")
        time.sleep(0.15)


def robot_policy(option="random", env=None):
     """
     A simple policy function that selects an action based on the specified option.
     Currently supports only a random policy.
     """
     if option == "random":
          return env.action_space.sample()  # Randomly select an action from the action space
     elif option == "round_robin":
          x, y = env.position
          # m = env.m
          n = env.n
          # Right
          if x % 2 == 0 and y < n - 1:
               return 3
          # Down
          elif x % 2 == 0 and y == n - 1:
               return 1
          # Left
          elif x % 2 == 1 and y > 0:
               return 2
          # Down
          else: return 1
     elif option == "priority_based":
          x, y = env.position
          m = env.m
          n = env.n
          cx, cy = (m - 1) /2 , (n - 1) / 2
          print (float(x), float(y))
          if x == 0 and y < n - 1:
               return 3
          elif float(x) <= cx and x - 1 <= y < n - 1 - x:
               return 3
          elif float(y) >= cy and n - 1 <= x + y < m + n - 2 - (n - 1 - y) * 2:
               return 1
          elif float(x) > cx and m - 1 < x + y <= m + n - 2 - (m - 1 - x) * 2:
               return 2
          elif float(y) < cy and x > y + 1 and x + y < m:
               return 0


if __name__ == "__main__":
    env = VacuumCleanerEnv(m= 10, n=10, obstacle=(4, 4))
    obs, _ = env.reset()
    x, y = env.position
    if env.dust_grid[x,y] == 1:
        env.total_reward += 1
        env.dust_grid[x,y] = 0
    env.render()
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = robot_policy(option="random", env=env)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}")
        if terminated or truncated:
            print("Episode finished with total reward:", env.total_reward)
            break