import gym
import matplotlib.pyplot as plt
import torch
import environment

from gym.wrappers.monitoring.video_recorder import VideoRecorder
from cnn import ExperienceReplay
from cnn import CNN
from agent import Agent
import numpy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('Mineline-v0')
video_recorder = VideoRecorder(env, './video.mp4')
#env.action_space.seed(42)
#print(env.action_space)
#print(env.observation_space)
stateervation = env.reset()
action_keys = ["attack", "left", "right"]
input_size = env.observation_space['pov'].shape   #taille de l'input de notre réseau
output_size = len(action_keys)    #taille de sortie

model = CNN(input_size, output_size)
exp_replay = ExperienceReplay()

def train():
    # define hyperparameters
    num_episodes = 100
    batch_size = 32
    eta = 0.001
    # create agent
    agent = Agent(env.action_space, env.observation_space, eta)
    # create experience replay
    # initialize lists
    rewards = []
    losses = []
    # iterate over episodes
    for i in range(num_episodes):
        # reset environment
        #state = agent.process_state(env.reset()['pov'])
        state = env.reset()
        state = torch.tensor(state['pov'].copy())
        #print(state.shape)
        #print(state)
        done = False
        total_reward = 0
        # iterate over steps
        while not done:
            # act
            action = agent.act(state, 0, done)
            # step
            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(next_state['pov'].copy())
            # save experience
            exp_replay.save(state, action, next_state, reward, done)
            # update state
            state = next_state
            # update total reward
            total_reward += reward
        # learn
        agent.learn(batch_size, exp_replay)
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

def test():
    # load model
    agent.cnn.load_state_dict(torch.load('model.pth'))
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
        # update state
        state = agent.process_state(next_state['pov'])
        # update total reward
        total_reward += reward
    # print
    print('Reward: {}'.format(total_reward))

