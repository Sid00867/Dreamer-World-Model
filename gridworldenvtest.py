from gridworldenv import * 
import time
import matplotlib.pyplot as plt


env = RLReadyEnv(
        env_kind= "simple",

        # Level params
        size = 10,

        # KeyCorridor difficulty params
        keycorridor_s= 5,   # room size S
        keycorridor_r= 5,   # number of rooms R

        tile_size= 4, # agent view - 7x7, tiling - 4x4, 4x7 = 28 so 28x28x3 images

        # Observation config
        obs_mode = "rgb",
        obs_scope = "partial",  #agent sight - 7x7

        # General options
        render_mode='human',
        agent_start_pos= (1, 1),
        agent_start_dir=0,
        max_steps= None,
        seed= 81,
)

def showimage(image):
        plt.imshow(image)
        plt.title(f"RGB Partial Obs (shape: {image.shape})")
        plt.axis('off')
        plt.show()

# core loop 
env.reset()
done = False
while not done:
    action = env.env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
#   showimage(obs['image'])
    env.render()
    time.sleep(0.1)




