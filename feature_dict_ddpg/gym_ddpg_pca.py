from struct import pack
import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from sklearn.decomposition import PCA
np.seterr(divide='ignore',invalid='ignore')

def get_sample_by_pca(list_data,batch_size,pca_list):
  # print(pca_list)
  pca_=np.array(pca_list)
  # print(pca_)
  sorted_index=np.lexsort([pca_[:,0]])
  # print('  sort   ->   ',sorted_index)
  memory_size=len(list_data)
  # print('batch_size   ',batch_size)
  dis=int(memory_size/batch_size)-1
  # print('dis      ',dis)
  random_idx=random.randint(0,dis)
  choose_data=[]
  for i in range(batch_size):
    choose_data.append(list_data[sorted_index[random_idx+i*dis]])
  
  return choose_data



class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, s):
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        s_dim = self.env.observation_space.shape[0]
        a_dim = self.env.action_space.shape[0]

        self.actor = Actor(s_dim, 256, a_dim)
        self.actor_target = Actor(s_dim, 256, a_dim)
        self.critic = Critic(s_dim+a_dim, 256, a_dim)
        self.critic_target = Critic(s_dim+a_dim, 256, a_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = self.critic_lr)
        #?????????
        self.buffer = []
        # ?????????
        self.pca_buffer=[]
        self.buffer_point=0
        self.capacity=10000
        self.batch_size=128
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
    def act(self, s0):
        s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0)
        a0 = self.actor(s0).squeeze(0).detach().numpy()
        return a0
    
    def put(self, *transition): 
        
        if len(self.buffer)< self.capacity:
          self.buffer.append(transition)
        else:
          self.buffer[self.buffer_point]=transition
        self.buffer_point+=1
        self.buffer_point=self.buffer_point%self.capacity

        pca = PCA(n_components=1)
        # print('transition  --->  ',np.array([transition[0]]))
        s_data=np.array([transition[0],[1,1]])
        low_dim_data = pca.fit_transform(s_data)
        # print(transition[0],'    low   ->   ',low_dim_data)
        if len(self.pca_buffer)< self.capacity:
          self.pca_buffer.append(low_dim_data[0])
        else:
          self.pca_buffer[self.buffer_point]=low_dim_data[0]

    
    def learn(self):
        if len(self.buffer) < self.batch_size:
          return 0
        
        #samples = random.sample(self.buffer, self.batch_size)
        samples = get_sample_by_pca(self.buffer,self.batch_size,self.pca_buffer)
        #print('samples  ->',samples)
        s0, a0, r1, s1 = zip(*samples)
        
        s0 = torch.tensor(s0, dtype=torch.float)
        a0 = torch.tensor(a0, dtype=torch.float)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size,-1)
        s1 = torch.tensor(s1, dtype=torch.float)
        
        def critic_learn():
            a1 = self.actor_target(s1).detach()
            y_true = r1 + self.gamma * self.critic_target(s1, a1).detach()
            
            y_pred = self.critic(s0, a0)
            
            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, y_true)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()
            
        def actor_learn():
            loss = -torch.mean( self.critic(s0, self.actor(s0)) )
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()
                                           
        def soft_update(net_target, net, tau):
            for target_param, param  in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
        critic_learn()
        actor_learn()
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)
        return 1
                                           
                                           
import time
ct = time.time()
# env = gym.make('Pendulum-v1')
env = gym.make('MountainCarContinuous-v0')

env.reset()
# env.render()

params = {
    'env': env,
    'gamma': 0.99, 
    'actor_lr': 0.001, 
    'critic_lr': 0.001,
    'tau': 0.02,
    'capacity': 10000, 
    'batch_size': 32,
    }
update_every = 50
agent = Agent(**params)
write=SummaryWriter('./gym/pca')
learn_num=0
for episode in range(100):
    s0 = env.reset()
    episode_reward = 0
    
    for step in range(500):
        # env.render()
        a0 = agent.act(s0)
        s1, r1, done, _ = env.step(a0)
        agent.put(s0, a0, r1, s1)

        episode_reward += r1 
        s0 = s1

        # if episode >= 5 and step % update_every == 0:
        #   for _ in range(update_every):
        learn_num+=agent.learn()
    write.add_scalar('reward',episode_reward,episode)
    print(episode, ': ', episode_reward)

print('pca  time  :',time.time()-ct)