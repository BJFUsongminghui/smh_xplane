import re
from unittest.mock import patch
import random
from pip import main
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

def getData(path):
  ea=event_accumulator.EventAccumulator(path)
  ea.Reload()
  print(ea.scalars.Keys())
  long_pos_reward=ea.scalars.Items('memory_size')
  print(len(long_pos_reward))
  return long_pos_reward

def pltFig(item1,item2,item3,reward1,reward2,reward3):
  
  # plt.title('Critic Loss')
  plt.title('Memory Size in episode')
  plt.plot(item3, reward3, color='mediumorchid', label='PER-DDPG',alpha=0.8)
  plt.plot(item2, reward2, color='darkgreen', label='PCA-DDPG',alpha=0.8)
  plt.plot(item1, reward1, color='tomato', label='DDPG',alpha=0.8)
  
  plt.legend()
  plt.xlabel('episode')
  plt.ylabel('memory_size')
  # plt.xlabel('step')
  # plt.ylabel('loss')
  plt.show()

def style(data):
  old_item=[]
  old_reward=[]
  len=1
  sum_loss=0
  for i in data:
    sum_loss+=i.value
    if (i.step+1)%len==0:
      old_item.append(i.step)
      old_reward.append(sum_loss/10.)
      sum_loss=0
  return old_item,old_reward
    
path1='./logs/ddpg/events.out.tfevents.1647840794.MacBook-Pro-3.local'
path2='./logs/f/events.out.tfevents.1647839933.MacBook-Pro-3.local'
path3='./logs/per/events.out.tfevents.1647839694.MacBook-Pro-3.local'

old_data=getData(path1)
new_item1,new_reward1=style(old_data)


old_data=getData(path2)
new_item2,new_reward2=style(old_data)

old_data=getData(path3)
new_item3,new_reward3=style(old_data)
pltFig(new_item1,new_item2,new_item3,new_reward1,new_reward2,new_reward3)
