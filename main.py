import gym
import matplotlib.pyplot as plt
import torch
from torch.distributed._shard.checkpoint import load_state_dict

from experienceReplay import ExperienceReplay
from qAgentCartPole1 import Agent
import numpy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('CartPole-v1', render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, video_folder="vid")
env.action_space.seed(42)


def randomGame():
    observation, info = env.reset(seed=42)
    x = []
    y = []
    nb_episodes = 1
    reward_cumul = 0

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

#randomGame()

def train():
    eta = 0.01
    batch_size = 128
    episodes = 300
    agent = Agent(env.action_space, env.observation_space, eta)
    experience = ExperienceReplay()

    rewardEp = numpy.array([])
    eps = numpy.array([])
    epcount = 0
    interactions = numpy.array([])

    for episode in range(episodes):
        observation, info = env.reset()
        reward_cumul = 0
        interactionsEp = 0
        while True:
            action = agent.act(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            reward_cumul += reward
            interactionsEp += 1
            experience.save(observation, action, next_observation, reward, terminated)
            observation = next_observation

            if terminated or truncated:
                rewardEp = numpy.append(rewardEp, reward_cumul)
                epcount += 1
                eps = numpy.append(eps, epcount)
                interactions = numpy.append(interactions, interactionsEp)
                print("Episode {} : reward = {}, epsilon = {}, eta = {}, loss = {}".format(episode, reward_cumul, agent.epsilon, agent.eta, agent.loss))
                break

            if (batch_size <= len(experience.buffer)):
                batch = experience.randomPick(batch_size)
                agent.learn(batch)

    env.close()
    torch.save(agent.dqn.model.state_dict(), "model.pth")

    plt.plot(eps, rewardEp)
    plt.title("ETA = {}".format(eta) + " batch_size = {}".format(batch_size))
    plt.ylabel("rewards")
    plt.xlabel("episodes")
    plt.show()

#train()

def testGame():
    x = []
    y = []
    nb_episodes = 1
    agent = Agent(env.action_space, env.observation_space, 0.01)
    agent.dqn.model.load_state_dict(torch.load("model.pth"))
    agent.dqn.model.eval()

    observation, info = env.reset()
    reward_cumul = 0
    for _ in range(100):
        while True:
            action = agent.act(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            reward_cumul += reward

            observation = next_observation
            if terminated or truncated:
                nb_episodes += 1
                x.append(nb_episodes)
                y.append(reward_cumul)
                print("Reward = {}".format(reward_cumul))
                reward_cumul = 0
                observation, info = env.reset()
                break

    env.close()
    plt.plot(x, y)
    plt.ylabel("rewards")
    plt.xlabel("episodes")
    plt.show()

testGame()