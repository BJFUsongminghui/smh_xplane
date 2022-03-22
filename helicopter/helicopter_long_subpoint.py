import numpy as np
import random

from gym import Env, spaces, wrappers
import time

SPACE_X=1000  # 
SPACE_Y=1000  # 
MAX_FUEL= 600  #
V_PLANE=2   #
A_PLANE=2

def get_min_range():
    return random.randrange(1,10)
def get_max_range():
    return random.randrange(SPACE_Y-10,SPACE_Y)
def get_all_range():
    return random.randrange(1,SPACE_Y)


class HelicopterSpace(Env):
    '''
    
    '''
    def __init__(self):
        super(HelicopterSpace, self).__init__()
        # Define a 2-D observation space
        self.observation_shape = (SPACE_X, SPACE_Y, 3)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape),
                                            high = np.ones(self.observation_shape),
                                            dtype=np.float16)
        # Define an action space range from 0 to 4
        self.action_space = spaces.Discrete(5,)
        self.max_fuel=MAX_FUEL
        # Create a canvas to render the environment images upon
        self.canvas = np.ones(self.observation_shape) * 1

        # Define elements present inside the environment
        self.elements = []

        # Permissible area of helicopter to be
        # 
        self.y_min = int(self.observation_shape[0] * 0.1)
        self.x_min = 0
        self.y_max = int(self.observation_shape[0] * 0.9)
        self.x_max = self.observation_shape[1]
        self.v_x=0
        self.v_y=0
        self.sub_point=[]

    def reset(self):
        '''
        Resets the environment to its initial state and returns the initial observation.
        '''
        # Reset the fuel consumed 
        self.fuel_left = self.max_fuel

        # Reset the reward
        self.ep_return = 0


        # Determine a place to initialise the helicopter
        # x = random.randrange(int(self.observation_shape[0] * 0.05), int(self.observation_shape[0] * 0.10))
        # y = random.randrange(int(self.observation_shape[1] * 0.15), int(self.observation_shape[1] * 0.20))
        x = SPACE_X/2
        y = SPACE_Y/2
        #   Initialise the helicopter
        self.helicopter = Helicopter('helicopter', self.x_max, self.x_min, self.y_max, self.y_min)
        self.helicopter.set_position(x, y)

        # Initialise the elements
        self.elements = [self.helicopter]
        # 
        self.spawned_fuel = Fuel("fuel", self.x_max, self.x_min, self.y_max, self.y_min)
        #

        choose=random.randint(0,3)
       
        fuel_x = get_min_range()
        fuel_y = get_all_range()
        if choose==1:
            fuel_x = get_all_range()
            fuel_y = get_min_range()
        
        if choose==2:
            fuel_x = get_max_range()
            fuel_y = get_all_range()

        if choose==3:
            fuel_x = get_all_range()
            fuel_y = get_max_range()
      
        self.spawned_fuel.set_position(fuel_x, fuel_y)

        # Append the spawned fuel tank to the elemetns currently present in the Env.
        self.elements.append(self.spawned_fuel)
        self.sub_point=[]
        self.get_sub_point(x,y,fuel_x,fuel_y)
        self.sub_point.append(self.spawned_fuel)
        # print(len(self.sub_point))
        # Reset the Canvas
        self.canvas = np.ones(self.observation_shape) * 1

        # Return the observation
        return self.get_state()

    def has_collided(self, elem1, elem2):
        '''
       
        '''
        x_col = False
        y_col = False


        elem1_x, elem1_y = elem1.get_position()
        elem2_x, elem2_y = elem2.get_position()

        if 2 * abs(elem1_x - elem2_x) <= (elem1.icon_w + elem2.icon_w):
            x_col = True

        if 2 * abs(elem1_y - elem2_y) <= (elem1.icon_h + elem2.icon_h):
            y_col = True

        if x_col and y_col:
            return True

        return False
    def assert_v(self,v):
        if v>V_PLANE:
            return V_PLANE
        if v<-1*V_PLANE:
            return -1*V_PLANE
        return v
    def step(self, action):
        '''
        Executes a step in the environment by applying an action.
        :param action: [action[0],action[1]]
        :return: new observation, reward, completion status, and other info.
        '''

        # Flag that marks the termination of an episode
        done = False
        # Decrease the fuel counter
        self.fuel_left -= 1
        # Reward for executing a step.
      
        #
        state_=self.get_state()
        self.v_x=self.assert_v(action[0]+self.v_x)
        self.v_y=self.assert_v(action[1]+self.v_y)

        self.helicopter.move(self.v_x, self.v_y)
        # d_x = self.get_state()[2]
        # d_y = self.get_state()[3]
        # reward=self.v_x/(d_x+1.111)+self.v_y/(d_y+1.111)
        # 
        state=self.get_state()
        reward = abs(state_[2]**2+state_[3]**2)-abs(state[2]**2+state[3]**2)

        
        if self.has_collided(self.sub_point[0], self.helicopter):
            reward +=(V_PLANE*2)
            if len(self.sub_point)==1:
                reward +=20
                done=True
            else:
                del(self.sub_point[0])
       
        # Increment the episodic return
        self.ep_return = reward
        
        # If out of fuel, end the episode.
        if self.fuel_left == 0:
            done = True

        return self.get_state(), reward, done, self.canvas

    def get_state(self):
      if len(self.sub_point)>0:
        return [self.v_x,self.v_y,(self.sub_point[0].x-self.helicopter.x),(self.sub_point[0].y-self.helicopter.y)]
      else:
        return [self.v_x,self.v_y,0.,0.]

    def get_cur_postion(self):
        return "plane pos ( "+ str(self.helicopter.x)+" - "+str(self.helicopter.y)+" )   target pos:( "+str(self.spawned_fuel.x)+" - "+str(self.spawned_fuel.y)+" )"
    def get_sub_point(self,plane_x,plane_y,point_x,point_y):
        x=point_x-plane_x
        y=point_y-plane_y
        sub_len=100
        if x<y:
            for i in range(int(y/sub_len)):
                i_y=sub_len*(1+i)
                i_x=sub_len*(1+i)*x/y
                sub_point_x=plane_x+i_x
                sub_point_y=plane_y+i_y
                sub_fuel=Fuel("fuel", self.x_max, self.x_min, self.y_max, self.y_min)
                sub_fuel.set_position(sub_point_x,sub_point_y)
                self.sub_point.append(sub_fuel)
        else:
            for i in range(int(x/sub_len)):
                i_x=sub_len*(1+i)
                i_y=sub_len*(1+i)*y/x
                sub_point_x=plane_x+i_x
                sub_point_y=plane_y+i_y
                sub_fuel=Fuel("fuel", self.x_max, self.x_min, self.y_max, self.y_min)
                sub_fuel.set_position(sub_point_x,sub_point_y)
                self.sub_point.append(sub_fuel)
        

class Point(object):
    '''
    elements
    '''
    def __init__(self,name,x_max,x_min,y_max,y_min):
        self.x = 0  # 
        self.y = 0
        self.x_max = x_max  # 
        self.x_min = x_min
        self.y_max = y_max
        self.y_min = y_min
        self.name = name

    def set_position(self,x,y):
        self.x = self.clamp(x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(y, self.y_min, self.y_max - self.icon_h)

    def get_position(self):
        return (self.x,self.y)

    def move(self,del_x, del_y):
        #  Move the points by certain value.
        self.x += del_x
        self.y += del_y
        self.set_position(self.x, self.y)

    def clamp(self,n,minn,maxn):
        '''
       
        '''
        return max(min(maxn,n),minn)


class Helicopter(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Helicopter, self).__init__(name, x_max, x_min, y_max, y_min) 
        self.icon_w = 16  # weight
        self.icon_h = 16  # height
  

class Bird(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Bird, self).__init__(name, x_max, x_min, y_max, y_min)
      
        self.icon_w = 8
        self.icon_h = 8


class Fuel(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Fuel, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon_w = 8
        self.icon_h = 8


import random

import DDPG as rl
import torch
from tensorboardX import SummaryWriter
import time
import numpy as np

MAX_EPISODES = 1000
MAX_EP_STEPS = 300
ON_TRAIN = True
MEMORY_CAPACITY_TRAIN = 500
MEMORY_CAPACITY = 200000

def assert_action(action):
    #return int(action*V_PLANE)
    return action*A_PLANE
def multi_run():
    xenv =  HelicopterSpace()
    s_dim = 4
    a_dim = 2
    ddpg = rl.DDPG(s_dim, a_dim)
    # ddpg=torch.load("1399_d.pl")
    s = xenv.reset()
    res_store_dir = "long_pos/logs/10000-sub1000-20v-10a" + time.strftime("-%Y-%m-%d-%H-%M-%S", time.localtime())
    writer = SummaryWriter( res_store_dir)  #
    train_step = 0

    for i in range(MAX_EPISODES):
        ep_r = 0
        # 
        j=0
        while(True):
            a = ddpg.choose_action(torch.FloatTensor(s)).detach().cpu().numpy()
            a=[assert_action(a[0]),assert_action(a[1])]
            s_, r, done,_ = xenv.step(a)
            
            # if (j + 1) % 10 == 0:
            #     print(xenv.get_cur_postion())
            #xenv.render()
            ep_r += r
            # if error != 4:
            ddpg.store_transition(s, a, r , s_, done)
            if ddpg.pointer > MEMORY_CAPACITY_TRAIN:
                loss_a, loss_c = ddpg.learn() 
                writer.add_scalar('critic loss', loss_c, global_step=train_step)
                writer.add_scalar('actor loss', loss_a, global_step=train_step)
                train_step += 1
            j+=1
            s = s_
            if done :
              print('Ep: %i | %s | ep_r: %.1f | steps: %i | memory:%.1f' % (i, '----' if not done else 'done', ep_r, j,ddpg.memory_size))
              s = xenv.reset()
              writer.add_scalar('long pos reward in eipsode', ep_r, global_step=i)
              writer.add_scalar('memory size', ddpg.memory_size, global_step=i)
              writer.add_scalar('steps in eipsode', j + 1, global_step=i)
              writer.add_scalar('every_reward in eipsode', ep_r / (j + 1), global_step=i)
              break

        if (i + 1) % 10 == 0:
            ddpg.noise_decay()

        # if (i + 1) % 200 == 0 and ddpg.pointer >= MEMORY_CAPACITY_TRAIN:
        #     torch.save(ddpg,"long_pos/models/"+str(i) + ".pl")


import math
def section(d):
    if d<100:
        return 1
    if d<200:
        return 2
    if d<300:
        return 3
    if d<400:
        return 4
    if d<500:
        return 5
    if d<600:
        return 6
    return 7
if __name__ == '__main__':
    multi_run()
    #tes()