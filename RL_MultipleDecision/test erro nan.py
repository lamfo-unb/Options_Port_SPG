# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 22:20:59 2018

@author: pedro
"""
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal

import numpy as np
import math

import utils
import model

import torch
print(torch.__version__)
print(torch.cuda.is_available())
print('Torch', torch.__version__, 'CUDA', torch.version.cuda)
print('Device:', torch.device('cuda:0'))
torch.cuda.is_available()

BATCH_SIZE = 1024
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001

xx=time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#ram.add(state, action, reward, new_state)
s1g,a1g,r1g,s2g = ram.sample(BATCH_SIZE)
# ---------------------- optimize critic ----------------------

		# Use target actor exploitation policy here for loss evaluation
for i in range(24):
    s1 = Variable(torch.from_numpy(s1g))
    a1 = Variable(torch.from_numpy(a1g[:,i].reshape(len(a1g),1)))
    r1 = Variable(torch.from_numpy(r1g[:,i].reshape(len(r1g),1)))
    s2 = Variable(torch.from_numpy(s2g))
#    s1 = (torch.from_numpy(s1g))
#    a1 = (torch.from_numpy(a1g[:,i].reshape(len(a1g),1)))
#    r1 = (torch.from_numpy(r1g[:,i].reshape(len(r1g),1)))
#    s2 = (torch.from_numpy(s2g))


    #[[actor,target_actor,actor_optimizer],[critic,target_critic,critic_optimizer]]
    target_actor=trainer.Jmodel[str(i)][0][1]
    target_critic=trainer.Jmodel[str(i)][1][1]
    critic=trainer.Jmodel[str(i)][1][0]
    actor=trainer.Jmodel[str(i)][0][0]
    actor_optimizer=trainer.Jmodel[str(i)][0][2]
    critic_optimizer=trainer.Jmodel[str(i)][1][2]

    if torch.cuda.is_available():
        s1,a1,r1,s2=s1.to(device),a1.to(device),r1.to(device),s2.to(device)
        target_actor=target_actor.to(device)
        target_critic=target_critic.to(device)
        critic=critic.to(device)
        actor=actor.to(device)
#        actor_optimizer.to(device)
#        critic_optimizer.to(device)


    a2 = target_actor.forward(s2).detach()
    next_val = torch.squeeze(target_critic.forward(s2, a2).detach())
    # y_exp = r + gamma*Q'( s2, pi'(s2))
    y_expected = torch.squeeze(r1) + (GAMMA*next_val)
    # y_pred = Q( s1, a1)
    y_predicted = torch.squeeze(critic.forward(s1, a1))

    # compute critic loss, and update the critic MSELoss
#    loss_critic = F.mse_loss(y_predicted, y_expected)
    loss_critic = F.mse_loss(y_predicted, y_expected)
    #		loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
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
    
    trainer.Jmodel.update({str(i):[[actor,target_actor,actor_optimizer],[critic,target_critic,critic_optimizer]]})
print((time.time()-xx))

		# if trainer.iter % 100 == 0:





from torch.distributions import Normal
action =np.array([])
		state = Variable(torch.from_numpy(state))
		for i in range(self.action_dim):
			mod=trainer.Jmodel[str(0)][0][0]
			probs=mod.forward(state.float())
			m = Normal(probs,0.05))
			smp=m.sample().detach().data.numpy()
			action =np.append(action,smp)
