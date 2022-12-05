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
    # define hyperparameters
    num_episodes = 100
    batch_size = 32
    eta = 0.001
    # create agent
    agent = Agent(env.action_space, env.observation_space, eta)
    # create experience replay
    exp_replay = ExperienceReplay()
    # initialize lists
    rewards = []
    losses = []
    # iterate over episodes
    for i in range(num_episodes):
        # reset environment
        state = agent.process_state(env.reset()['pov'])
        done = False
        total_reward = 0
        # iterate over steps
        while not done:
            # act
            action = agent.act(state, 0, done)
            # step
            next_state, reward, done, _ = env.step(action)
            # save experience
            exp_replay.save(state, action, next_state, reward, done)
            # update state
            state = next_state
            # update total reward
            total_reward += reward
        # sample batch
        batch = exp_replay.randomPick(batch_size)
        # learn
        agent.learn(batch)
        # update lists
        rewards.append(total_reward)
        losses.append(agent.loss)
        # print
        print('Episode: {}, Reward: {}, Loss: {}'.format(i, total_reward, agent.loss))
        # plot
        plt.plot(rewards)
        plt.plot(losses)
        plt.show()
        # save model
        torch.save(agent.cnn.state_dict(), 'model.pth')

train()