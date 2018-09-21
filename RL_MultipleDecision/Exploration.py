# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 19:28:08 2018

@author: pedro
"""

from __future__ import division
import gym
import numpy as np
import torch
from torch.autograd import Variable
import os
import psutil
import gc
import sys

sys.path.insert(0,'C:\\Users\\pedro\\OneDrive\\Documentos\\GitHub\\PyTorch-ActorCriticRL-master')

#import train
import buffer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math

import utils
#import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPS = 0.003

def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)

class Critic(nn.Module):

	def __init__(self, state_dim, action_dim):
		"""
		:param state_dim: Dimension of input state (int)
		:param action_dim: Dimension of input action (int)
		:return:
		"""
		super(Critic, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim

		self.fcs1 = nn.Linear(state_dim,256)
		self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
		self.fcs2 = nn.Linear(256,128)
		self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())

		self.fca1 = nn.Linear(action_dim,128)
		self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

		self.fc2 = nn.Linear(256,128)
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

		self.fc3 = nn.Linear(128,1)
		self.fc3.weight.data.uniform_(-EPS,EPS)

	def forward(self, state, action):
		"""
		returns Value function Q(s,a) obtained from critic network
		:param state: Input state (Torch Variable : [n,state_dim] )
		:param action: Input Action (Torch Variable : [n,action_dim] )
		:return: Value function : Q(S,a) (Torch Variable : [n,1] )
		"""
		s1 = F.relu(self.fcs1(state.float()))
		s2 = F.relu(self.fcs2(s1))
		a1 = F.relu(self.fca1(action))
		x = torch.cat((s2,a1),dim=1)

		x = F.relu(self.fc2(x))
		x = self.fc3(x)

		return x


class Actor(nn.Module):

	def __init__(self, state_dim, action_dim, action_lim):
		"""
		:param state_dim: Dimension of input state (int)
		:param action_dim: Dimension of output action (int)
		:param action_lim: Used to limit action in [-action_lim,action_lim]
		:return:
		"""
		super(Actor, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_lim = action_lim

		self.fc1 = nn.Linear(state_dim,256)
		self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

		self.fc2 = nn.Linear(256,128)
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

		self.fc3 = nn.Linear(128,64)
		self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

		self.fc4 = nn.Linear(64,action_dim)
		self.fc4.weight.data.uniform_(-EPS,EPS)

	def forward(self, state):
		"""
		returns policy function Pi(s) obtained from actor network
		this function is a gaussian prob distribution for all actions
		with mean lying in (-1,1) and sigma lying in (0,1)
		The sampled action can , then later be rescaled
		:param state: Input state (Torch Variable : [n,state_dim] )
		:return: Output action (Torch Variable: [n,action_dim] )
		"""
		x = F.relu(self.fc1(state.float()))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		action = F.tanh(self.fc4(x))

		action = action * torch.tensor(np.array(self.action_lim))
		return action

def get_exploitation_action(state):
    """
    gets the action from target actor added with exploration noise
    :param state: state (Numpy array)
    :return: sampled action (Numpy array)
    """
    state = Variable(torch.from_numpy(state))
    action = target_actor.forward(state.float()).detach()
    return action.data.numpy()

def get_exploration_action(state):
    """
    gets the action from actor added with exploration noise
    :param state: state (Numpy array)
    :return: sampled action (Numpy array)
    """
    state = Variable(torch.from_numpy(state))
    action = actor.forward(state.float()).detach()
    new_action = action.data.numpy() + (noise.sample() * action_lim)
    return new_action

def optimize():
    """
    Samples a random batch from replay memory and performs optimization
    :return:
    """
    s1,a1,r1,s2 = ram.sample(BATCH_SIZE)
    
    s1 = Variable(torch.from_numpy(s1))
    a1 = Variable(torch.from_numpy(a1))
    r1 = Variable(torch.from_numpy(r1))
    s2 = Variable(torch.from_numpy(s2))
    
    # ---------------------- optimize critic ----------------------
    # Use target actor exploitation policy here for loss evaluation
    a2 = target_actor.forward(s2).detach()
    next_val = torch.squeeze(target_critic.forward(s2, a2).detach())
    # y_exp = r + gamma*Q'( s2, pi'(s2))
    y_expected = r1 + GAMMA*next_val
    # y_pred = Q( s1, a1)
    y_predicted = torch.squeeze(critic.forward(s1, a1))
    # compute critic loss, and update the critic
    loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
    critic_optimizer.zero_grad()
    loss_critic.backward()
    critic_optimizer.step()
    
    # ---------------------- optimize actor ----------------------
    pred_a1 = actor.forward(s1)
    loss_actor = -1*torch.sum(critic.forward(s1, pred_a1))
    actor_optimizer.zero_grad()
    loss_actor.backward()
    actor_optimizer.step()
    
    utils.soft_update(target_actor, actor, TAU)
    utils.soft_update(target_critic, critic, TAU)
    
    # if iter % 100 == 0:
    # print 'Iteration :- ', iter, ' Loss_actor :- ', loss_actor.data.numpy(),
    # ' Loss_critic :- ', loss_critic.data.numpy()
    # iter += 1
    
def save_models(episode_count):
    """
    saves the target actor and critic models
    :param episode_count: the count of episodes iterated
    :return:
    """
    torch.save(target_actor.state_dict(), './Models/' + str(episode_count) + '_actor.pt')
    torch.save(target_critic.state_dict(), './Models/' + str(episode_count) + '_critic.pt')
    print ('Models saved successfully')
    
def load_models(episode):
    """
    loads the target actor and critic models, and copies them onto actor and critic models
    :param episode: the count of episodes iterated (used to find the file name)
    :return:
    """
    actor.load_state_dict(torch.load('./Models/' + str(episode) + '_actor.pt'))
    critic.load_state_dict(torch.load('./Models/' + str(episode) + '_critic.pt'))
    utils.hard_update(target_actor, actor)
    utils.hard_update(target_critic, critic)
    print ('Models loaded succesfully')
    
        
env = gym.make('MountainCarContinuous-v0')
#env = gym.make('MountainCar-v0')
#env = gym.make('Pendulum-v0')
#env = gym.make('CartPole-v0')
#env = gym.make('LunarLanderContinuous-v2')
#env = gym.make('PongDeterministic-v0')

MAX_EPISODES = 5000
MAX_STEPS = 1000
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]

print (' State Dimensions :- ', S_DIM)
print (' Action Dimensions :- ', A_DIM)
print (' Action Max :- ', A_MAX)

ram = buffer.MemoryBuffer(MAX_BUFFER)
#trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)


BATCH_SIZE = 128
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001

state_dim, action_dim, action_lim, ram=S_DIM, A_DIM, A_MAX, ram

iiter = 0

noise = utils.OrnsteinUhlenbeckActionNoise(action_dim)

actor = Actor(state_dim, action_dim, action_lim)
target_actor = Actor(state_dim, action_dim, action_lim)
actor_optimizer = torch.optim.Adam(actor.parameters(),LEARNING_RATE)

critic = Critic(state_dim, action_dim)
target_critic = Critic(state_dim, action_dim)
critic_optimizer = torch.optim.Adam(critic.parameters(),LEARNING_RATE)

utils.hard_update(target_actor, actor)
utils.hard_update(target_critic, critic)


for _ep in range(MAX_EPISODES):
	observation = env.reset()
	print ('EPISODE :- ', _ep)
	for r in range(MAX_STEPS):
		state = np.float32(observation)
#		action = get_exploration_action(state)

		if _ep%10 == 0:
		 	# validate every 5th episode
			action = get_exploitation_action(state)
		else:
		 	# get action based on observation, use exploration policy here
			action = get_exploration_action(state)

		if _ep%100 == 0:
			env.render()

		new_observation, reward, done, info = env.step(action)

#		reward=reward+((new_observation[0]+1.21)/1.81)/100
		# # dont update if this is validation
		# if _ep%50 == 0 or _ep>450:
		# 	continue
		if done:
			new_state = None
		else:
			new_state = np.float32(new_observation)
			# push this exp in ram
			ram.add(state, action, reward, new_state,done)

		observation = new_observation

		# perform optimization
		optimize()
		if done:
			print(r)
			r=0
			break
 	# check memory consumption and clear memory
	gc.collect()
	# process = psutil.Process(os.getpid())
	# print(process.memory_info().rss)

	if _ep%100 == 0:
		save_models(_ep)


print ('Completed episodes')


#rw=[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
#ad=0
#
#for k in reversed(range(len(rw))):
#    ad=ad*GAMMA+rw[k]
#    rw[k]=ad
#
#[i for i in rw]
#
#for j in reversed(range(len(reward_poolSUPER[i]))):
#    running_add = running_add * gamma + reward_poolSUPER[i][j]
#    reward_poolSUPER[i][j] = running_add
#
#
#
