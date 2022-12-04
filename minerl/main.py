import gym
import matplotlib.pyplot as plt
import torch
import environment

from gym.wrappers.monitoring.video_recorder import VideoRecorder
from cnn import ExperienceReplay
from agent import Agent
import numpy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('Mineline-v0')
#env = VideoRecorder(env, './video')
#env.action_space.seed(42)
print(env.action_space)
print(env.observation_space)


def train():
    eta = 0.01
    batch_size = 64
    episodes = 300
    agent = Agent(env.action_space, env.observation_space, eta)
    experience = ExperienceReplay()

    rewardEp = numpy.array([])
    eps = numpy.array([])
    epcount = 0
    interactions = numpy.array([])

    for episode in range(episodes):
        observation = env.reset()['pov']
        print(observation)
        reward_cumul = 0
        interactionsEp = 0
        while True:
            action = agent.act(numpy.array(observation))
            next_observation, reward, done, info = env.step(action)
            reward_cumul += reward
            interactionsEp += 1
            experience.save(observation, action, next_observation, reward, done)
            observation = next_observation

            if done:
                rewardEp = numpy.append(rewardEp, reward_cumul)
                epcount += 1
                eps = numpy.append(eps, epcount)
                interactions = numpy.append(interactions, interactionsEp)
                print("Episode {} : reward = {}, epsilon = {}, eta = {}, loss = {}".format(episode, reward_cumul, agent.epsilon, agent.eta, agent.loss))
                break

            if (batch_size <= len(experience.buffer)):
                batch = experience.randomPick(batch_size)
                agent.learn(batch)
        agent.hard_update()

    env.close()
    torch.save(agent.cnn.state_dict(), "model.pth")

    plt.plot(eps, rewardEp)
    plt.title("ETA = {}".format(eta) + " batch_size = {}".format(batch_size))
    plt.ylabel("rewards")
    plt.xlabel("episodes")
    plt.show()

train()