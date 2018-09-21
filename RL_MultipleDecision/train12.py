from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal

import numpy as np
import math

import utils
import model

BATCH_SIZE = 128*2*2
LEARNING_RATE = 0.0001
GAMMA = 0.99
TAU = 0.0001



class Trainer:

	def __init__(self, state_dim, action_dim, action_lim, ram):
		"""
		:param state_dim: Dimensions of state (int)
		:param action_dim: Dimension of action (int)
		:param action_lim: Used to limit action in [-action_lim,action_lim]
		:param ram: replay memory buffer object
		:return:
		"""
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_lim = action_lim
		self.ram = ram
		self.iter = 0
		self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim)

		if torch.cuda.is_available():
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.Jmodel={}
		for i in range(self.action_dim):
			actor = model.Actor(self.state_dim,1, self.action_lim)
			target_actor = model.Actor(self.state_dim, 1, self.action_lim)
			actor_optimizer = torch.optim.Adam(actor.parameters(),LEARNING_RATE)
#			actor_optimizer = torch.optim.Adam(actor.parameters(),LEARNING_RATE)

			critic = model.Critic(self.state_dim, 1)
			target_critic = model.Critic(self.state_dim, 1)
			critic_optimizer = torch.optim.Adam(critic.parameters(),LEARNING_RATE)
#			critic_optimizer = torch.optim.Adam(critic.parameters(),LEARNING_RATE)

			utils.hard_update(target_actor, actor)
			utils.hard_update(target_critic, critic)
			self.Jmodel.update({str(i):[[actor,target_actor,actor_optimizer],[critic,target_critic,critic_optimizer]]})

	def get_exploitation_action(self, state):
		"""
		gets the action from target actor added with exploration noise
		:param state: state (Numpy array)
		:return: sampled action (Numpy array)
		"""
		# if torch.cuda.is_available():
		# 	self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		action=np.array([])
		state = Variable(torch.from_numpy(state))
		for i in range(self.action_dim):
			mod=self.Jmodel[str(i)][0][0]
			mod=mod.to('cpu')
			action =np.append(action,(mod.forward(state.float()).detach().data.numpy()))

		return action

	def get_exploration_action(self, state):
		"""
		gets the action from actor added with exploration noise
		:param state: state (Numpy array)
		:return: sampled action (Numpy array)
		"""
		# if torch.cuda.is_available():
		# 	self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		action=np.array([])
		state = Variable(torch.from_numpy(state))
		for i in range(self.action_dim):
			mod=self.Jmodel[str(i)][0][0]
			# if torch.cuda.is_available():
			mod=mod.to('cpu')
			probs=mod.forward(state.float())
			m = Normal(probs,abs(probs)*0.2)
			# m = Normal(probs,1/24*0.15)
			smp=m.sample().detach().data.numpy()
			action =np.append(action,smp)
		new_action = action + (self.noise.sample() * self.action_lim)
		return new_action

	def optimize(self):
		"""
		Samples a random batch from replay memory and performs optimization
		:return:
		"""
		if torch.cuda.is_available():
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #ram.add(state, action, reward, new_state)
		s1g,a1g,r1g,s2g = self.ram.sample(BATCH_SIZE)
        # ---------------------- optimize critic ----------------------
		# Use target actor exploitation policy here for loss evaluation
		for i in range(self.action_dim):
			s1 = Variable(torch.from_numpy(s1g))
			a1 = Variable(torch.from_numpy(a1g[:,i].reshape(len(a1g),1)))
			r1 = Variable(torch.from_numpy(r1g[:,i].reshape(len(r1g),1)))
			s2 = Variable(torch.from_numpy(s2g))
			
			#[[actor,target_actor,actor_optimizer],[critic,target_critic,critic_optimizer]]
			target_actor=self.Jmodel[str(i)][0][1]
			target_critic=self.Jmodel[str(i)][1][1]
			critic=self.Jmodel[str(i)][1][0]
			actor=self.Jmodel[str(i)][0][0]
			actor_optimizer=self.Jmodel[str(i)][0][2]
			critic_optimizer=self.Jmodel[str(i)][1][2]
			
			if torch.cuda.is_available():
				s1,a1,r1,s2=s1.to(self.device),a1.to(self.device),r1.to(self.device),s2.to(self.device)
				target_actor=target_actor.to(self.device)
				target_critic=target_critic.to(self.device)
				critic=critic.to(self.device)
				actor=actor.to(self.device)
	#        	actor_optimizer.to(device)
       		#	critic_optimizer.to(device)

			a2 = target_actor.forward(s2).detach()
			next_val = torch.squeeze(target_critic.forward(s2, a2).detach())
			# y_exp = r + gamma*Q'( s2, pi'(s2))
			y_expected = torch.squeeze(r1) + GAMMA*next_val
			# y_pred = Q( s1, a1)
			y_predicted = torch.squeeze(critic.forward(s1, a1))
    
			# compute critic loss, and update the critic MSELoss
			# loss_critic = F.mse_loss(y_predicted, y_expected)
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

			if torch.cuda.is_available():
				s1,a1,r1,s2=s1.to("cpu"),a1.to("cpu"),r1.to("cpu"),s2.to("cpu")
				target_actor=target_actor.to("cpu")
				target_critic=target_critic.to("cpu")
				critic=critic.to("cpu")
				actor=actor.to("cpu")

			self.Jmodel.update({str(i):[[actor,target_actor,actor_optimizer],[critic,target_critic,critic_optimizer]]})

		# if self.iter % 100 == 0:		# if self.iter % 100 == 0:
		# 	print 'Iteration :- ', self.iter, ' Loss_actor :- ', loss_actor.data.numpy(),\
		# 		' Loss_critic :- ', loss_critic.data.numpy()
		# self.iter += 1

	def save_models(self, episode_count):
		"""
		saves the target actor and critic models
		:param episode_count: the count of episodes iterated
		:return:
		"""
		for i in range(self.action_dim):
			target_actor=self.Jmodel[str(i)][0][1]
			target_critic=self.Jmodel[str(i)][1][1]
			torch.save(target_actor.state_dict(), './Models/' + str(episode_count) + str(i)+'_actor.pt')
			torch.save(target_critic.state_dict(), './Models/' + str(episode_count) + str(i)+'_critic.pt')
		print ('Models saved successfully')

	def load_models(self, episode):
		"""
		loads the target actor and critic models, and copies them onto actor and critic models
		:param episode: the count of episodes iterated (used to find the file name)
		:return:
		"""
		for i in range(self.action_dim):
			target_actor=self.Jmodel[str(i)][0][1]
			target_critic=self.Jmodel[str(i)][1][1]
		
			actor.load_state_dict(torch.load('./Models/' + str(episode) + str(i)+ '_actor.pt'))
			critic.load_state_dict(torch.load('./Models/' + str(episode) + str(i)+ '_critic.pt'))
			utils.hard_update(target_actor,actor)
			utils.hard_update(target_critic,critic)
			self.Jmodel.update({str(i):[[actor,target_actor,actor_optimizer],[critic,target_critic,critic_optimizer]]})
		
		print ('Models loaded succesfully')
