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
  long_pos_reward=ea.scalars.Items('critic_loss')
  print(len(long_pos_reward))
  return long_pos_reward

def pltFig(item1,item2,item3,item4,reward1,reward2,reward3,reward4):
  
  plt.title('Critic Loss')
  # plt.title('Actor Loss')
  # plt.title('Memory Size in episode')
  plt.plot(item3, reward3, color='mediumorchid', label='Feature-3L-6M',alpha=0.9)
  plt.plot(item2, reward2, color='darkgreen', label='Feature-3L-sub5-6M',alpha=0.9)
  plt.plot(item1, reward1, color='tomato', label='DDPG-3L-6M-sub5',alpha=0.9)
  plt.plot(item4, reward4, color='dodgerblue', label='Feature-3L-3M-sub5',alpha=0.9)
  
  plt.legend()
  plt.xlabel('step')
  plt.ylabel('loss')
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
      old_reward.append(sum_loss)
      sum_loss=0
  return old_item,old_reward
def style_(data):
  old_item=[]
  old_reward=[]
  len=1
  sum_loss=0
  for i in data:
    sum_loss+=i.value
    if (i.step+1)%len==0:
      old_item.append(i.step)
      old_reward.append(sum_loss/1000.)
      sum_loss=0
  return old_item,old_reward
    
# path1='./PER-3L-6M/events.out.tfevents.1648093014.MacBook-Pro-3.local'
# path2='./Feature-3L-sub5-6M/events.out.tfevents.1648104928.MacBook-Pro-3.local'
# path3='./DDPG-3L-6M/events.out.tfevents.1648103688.MacBook-Pro-3.local'
path1='./ablation_experiment/DDPG-3L-6M-sub5/events.out.tfevents.1648102459.MacBook-Pro-3.local'

path2='./Feature-3L-sub5-6M/events.out.tfevents.1648104928.MacBook-Pro-3.local'
path3='./ablation_experiment/Feature-3L-6M/events.out.tfevents.1648102430.MacBook-Pro-3.local'
path4='./ablation_experiment/Feature-3L-sub5-3M/events.out.tfevents.1648093676.MacBook-Pro-3.local'
old_data=getData(path1)
new_item1,new_reward1=style(old_data)


old_data=getData(path2)
new_item2,new_reward2=style(old_data)

old_data=getData(path3)
new_item3,new_reward3=style(old_data)

old_data=getData(path4)
new_item4,new_reward4=style(old_data)
pltFig(new_item1,new_item2,new_item3,new_item4,new_reward1,new_reward2,new_reward3,new_reward4)
