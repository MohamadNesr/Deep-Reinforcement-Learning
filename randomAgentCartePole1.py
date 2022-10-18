import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1', render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, video_folder="vid")
env.action_space.seed(42)
nb_episodes = 1
reward_cumul = 0

observation, info = env.reset(seed=42)
x = []
y = []

for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    reward_cumul += reward

    if terminated or truncated:
        nb_episodes += 1
        x.append(nb_episodes)
        y.append(reward_cumul)
        reward_cumul = 0
        observation, info = env.reset()
env.close()

plt.plot(x, y)
plt.ylabel("rewards")
plt.xlabel("episodes")
plt.show()

