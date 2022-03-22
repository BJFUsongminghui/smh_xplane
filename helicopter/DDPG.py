import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import time
from sklearn.decomposition import PCA
#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.0001  # learning rate for actor
LR_C = 0.0002  # learning rate for critic
GAMMA = 0.99  # reward discount
TAU = 0.005  # soft replacement
MEMORY_CAPACITY = 30000
BATCH_SIZE = 256

RENDER = False
# edited by wsr 2021-5-31，使用GPU计算
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')  # 训练的设备
Explore_Noise = 0.1  # 探索的噪声

###############################  DDPG  ####################################
def getFeature(s,a):
    # print([1]*4)
    pca = PCA(n_components=1)
    # print('transition  --->  ',np.array([transition[0]]))
    s_data=np.array([s,[1,1,1,1]])
    # print('s_',s_data)
    low_dim_data = pca.fit_transform(s_data)
    a_data=np.array([a,[1,1]])
    low_dim_a = pca.fit_transform(a_data)
    cur_key=str(round(low_dim_data[0][0],3))+'_'+str(round(low_dim_a[0][0],3))
    return cur_key
class ANet(nn.Module):  # ae(s)=a
    def __init__(self, s_dim, a_dim):  # edited by wsr 2021-5-31，进一步减小网络规模
        super(ANet, self).__init__()
        self.fc1 = nn.Linear(s_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc7 = nn.Linear(256, 128)
        self.out = nn.Linear(128, a_dim)

    def forward(self, x):
        a_1 = F.relu(self.fc1(x))
        a_2 = F.relu(self.fc2(a_1))
        a_7 = F.relu(self.fc7(a_2))
        a_8 = self.out(a_7)
        a_out = torch.tanh(a_8)
        actions_value = a_out
        return actions_value


class CNet(nn.Module):  # ae(s)=a
    def __init__(self, s_dim, a_dim):  # edited by wsr 2021-5-31，进一步减小网络规模
        super(CNet, self).__init__()
        self.fcs = nn.Linear(s_dim + a_dim, 128)  # edited by wsr 2021-5-31，变更网络输入
        # self.fca = nn.Linear(a_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc7 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, s, a):    # edited by wsr 2021-5-31，变更网络计算，先拼接
        critic_input = torch.cat((s, a), dim=-1).float()
        x = F.relu(self.fcs(critic_input))
        net_1 = F.relu(self.fc2(x))
        net = F.relu(self.fc7(net_1))
        actions_value = self.out(net)
        return actions_value


class DDPG(object):
    def __init__(self, s_dim, a_dim):
        self.a_dim, self.s_dim = a_dim, s_dim
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1 + 1), dtype=np.float32)
        self.pointer = 0

        # self.sess = tf.Session()
        self.Actor_eval = ANet(s_dim, a_dim).to(DEVICE)
        self.Actor_target = ANet(s_dim, a_dim).to(DEVICE)
        self.Critic_eval = CNet(s_dim, a_dim).to(DEVICE)
        self.Critic_target = CNet(s_dim, a_dim).to(DEVICE)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr=LR_A)
        self.loss_td = nn.MSELoss()

        # edited by wsr 2021-5-31，评估和目标网络初始参数一致
        self.Actor_target.load_state_dict(self.Actor_eval.state_dict())
        self.Critic_target.load_state_dict(self.Critic_eval.state_dict())

        self.noise = Explore_Noise
        self.table=dict()
        self.memory_size=0

    def choose_action(self, s, explore=True):
        if explore:
            action = torch.tanh(self.Actor_eval(s))
            action = action + self._sample_exploration_noise(action)
            # action = torch.clamp(action, 0-ACTION_EDGE, ACTION_EDGE)
        else:
            action = torch.tanh(self.Actor_eval(s)).detach()
        return action

    def _sample_exploration_noise(self, action):
        mean = torch.zeros(action.size()).to(DEVICE)
        var = torch.ones(action.size()).to(DEVICE)
        return torch.normal(mean, self.noise * var)

    def learn(self):

        for x in self.Actor_target.state_dict().keys():
            eval('self.Actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Actor_target.' + x + '.data.add_(TAU*self.Actor_eval.' + x + '.data)')
        for x in self.Critic_target.state_dict().keys():
            eval('self.Critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Critic_target.' + x + '.data.add_(TAU*self.Critic_eval.' + x + '.data)')

        # soft target replacement
        # self.sess.run(self.soft_replace)  # 用ae、ce更新at，ct
        if self.pointer > MEMORY_CAPACITY:
            indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        else:
            indices = np.random.choice(self.pointer, size=BATCH_SIZE)

        bt = self.memory[indices, :]
        bs = torch.FloatTensor(bt[:, :self.s_dim]).to(DEVICE)
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim]).to(DEVICE)
        br = torch.FloatTensor(bt[:, -self.s_dim - 2: -self.s_dim-1]).to(DEVICE)
        bs_ = torch.FloatTensor(bt[:, -self.s_dim - 1: -1]).to(DEVICE)
        bdone = torch.FloatTensor(bt[:, -1]).view(BATCH_SIZE, 1).to(DEVICE) # edited by wsr 2021-5-31，加入最终状态标示

        a = self.choose_action(bs)
        q = self.Critic_eval(bs, a)  # loss=-q=-ce（s,ae（s））更新ae   ae（s）=a   ae（s_）=a_
        # 如果 a是一个正确的行为的话，那么它的Q应该更贴近0
        loss_a = -torch.mean(q)
        # print(q)
        # print(loss_a)
        self.atrain.zero_grad()
        loss_a.backward()
        nn.utils.clip_grad_norm_(self.Actor_eval.parameters(), 0.5)
        self.atrain.step()

        a_ = self.choose_action(bs_, explore=False)  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        q_ = self.Critic_target(bs_, a_)  # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        q_target = (br + GAMMA * q_ * (1 - bdone)).detach().float()  # q_target = 负的acacacacacacac
        # print(q_target)
        q_v = self.Critic_eval(bs, ba).float()
        # print(q_v)
        td_error = self.loss_td(q_target, q_v)
        # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,但这个ae(s)是记忆中的ba，让ce得出的Q靠近Q_target,让评价更准确
        # print(td_error)
        self.ctrain.zero_grad()
        td_error.backward()
        nn.utils.clip_grad_norm_(self.Critic_eval.parameters(), 0.5)
        self.ctrain.step()

        return loss_a.detach().cpu().numpy(), td_error.detach().cpu().numpy()

    def store_transition(self, s, a, r, s_, done=False):
        # if self.hasInBuffer(s,a):
        #     return 
        self.memory_size+=1
        if self.memory_size>= MEMORY_CAPACITY-1:
            self.memory_size=MEMORY_CAPACITY
            # self.deleteTable(self.memory[self.pointer, :][0],self.memory[self.pointer, :][1])
        transition = np.hstack((s, a, [r], s_, [done]))
        self.memory[self.pointer, :] = transition
        self.pointer += 1
        self.pointer  = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        # self.addTable(s,a)

    def hasInBuffer(self,s,a):
    #   print('s   ',s,'     a    ',a)
    
      cur_key=getFeature(s,a)
      if cur_key in self.table:
        return True
      else:
        return False
    def addTable(self,s,a):
      cur_key=getFeature(s,a)
      index = self.pointer % MEMORY_CAPACITY
      self.table[cur_key]=index
      
    def deleteTable(self,s,a):
      cur_key=getFeature(s,a)
      del self.table[cur_key]

    def noise_decay(self):
        self.noise = self.noise * 0.995

