import numpy as np

class MovingSquareEnv:
    def __init__(self):
        self.size = 28
        self.agent_pos = np.array([5, 14])
        self.goal_pos = np.array([22, 14])
        self.max_steps = 50
        self.current_step = 0
        
        # Action Map: 0:Up, 1:Down, 2:Left, 3:Right
        self.action_map = {
            0: np.array([-1, 0]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([0, 1]),
        }

    def reset(self):
        self.current_step = 0
        # Reset to fixed positions for easiest learning
        self.agent_pos = np.array([5, 14]) 
        self.goal_pos = np.array([22, 14]) 
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # Move Agent
        move = self.action_map.get(action, np.array([0, 0]))
        self.agent_pos = np.clip(self.agent_pos + move, 2, self.size - 3) # Keep in bounds

        # Calc Reward
        dist = np.abs(self.agent_pos - self.goal_pos).sum()
        
        done = False
        reached_goal = False
        
        if dist < 2: # Touching
            reward = 10.0
            done = True
            reached_goal = True
        else:
            reward = -0.1 # Slight time penalty
            
        if self.current_step >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done, False, {}, reached_goal

    def _get_obs(self):
        # Create Black Background (28x28x3)
        img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        
        # Draw Red Goal (3x3)
        gy, gx = self.goal_pos
        img[gy-1:gy+2, gx-1:gx+2, 0] = 255 
        
        # Draw Blue Agent (3x3)
        ay, ax = self.agent_pos
        img[ay-1:ay+2, ax-1:ax+2, 2] = 255
        
        # Dictionary format to match your code's expectation
        return {'image': img}

    def render(self):
        # Just return the image for visualization
        return self._get_obs()['image']