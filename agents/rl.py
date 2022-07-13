from stable_baselines3 import PPO
from gym_game.envs.drone_env import *
import time

# This class represents the training and testing of our rl agent
# We used this to create our rl agent: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
# If you want to train the agent then train = True, and if you want to test the agent then train = false (line 9)

train = False

if train:
    env = gym.make('Pygame-v0')

    model = PPO("MlpPolicy", env, verbose=1)

    model.learn(total_timesteps=2000000)
    model.save('1100-2')
    print("training completed")
else:

    env = gym.make('Pygame-v0')
    model = PPO.load("final-7.zip")
    model.set_env(env)
    random_seed = int(time.time())
    model.set_random_seed(random_seed)
    obs = env.reset()

    try:
        while True:
            env.render()

            action, _states = model.predict(obs)

            obs, reward, done, _ = env.step(action)

            if done is True:
                state = env.reset()
    finally:
        env.close()


