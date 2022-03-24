import re
from unittest.mock import patch
import random
from pip import main
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

def getMemorySize(path):
  ea=event_accumulator.EventAccumulator(path)
  ea.Reload()
  print(ea.scalars.Keys())
  long_pos_reward=ea.scalars.Items('memory_size')
  print(len(long_pos_reward))
  return long_pos_reward

def pltMemorySizeFig(item1,item2,item3,reward1,reward2,reward3):
  
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

def styleMemorySize(data):
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

def getLoss(path,loss):
  ea=event_accumulator.EventAccumulator(path)
  ea.Reload()
  print(ea.scalars.Keys())
  long_pos_reward=ea.scalars.Items(loss)
  print(len(long_pos_reward))
  return long_pos_reward

def pltLossFig(item1,item2,item3,reward1,reward2,reward3,title):
  
  # plt.title('Critic Loss')
  plt.title(title)
  plt.plot(item3, reward3, color='mediumorchid', label='PER-DDPG',alpha=0.9)
  plt.plot(item2, reward2, color='darkgreen', label='PCA-DDPG',alpha=0.9)
  plt.plot(item1, reward1, color='tomato', label='DDPG',alpha=0.9)
  
  plt.legend()
  plt.xlabel('step')
  plt.ylabel('loss')
  # plt.xlabel('step')
  # plt.ylabel('loss')
  plt.show()

def styleLoss(data):
  old_item=[]
  old_reward=[]
  len=10
  sum_loss=0
  for i in data:
    sum_loss+=i.value
    if (i.step+1)%len==0:
      old_item.append(i.step)
      old_reward.append(sum_loss/10.)
      sum_loss=0
  return old_item,old_reward
    
def pltRewardFig(item1,item2,item3,reward1,reward2,reward3,title):
  
  # plt.title('Critic Loss')
  plt.title(title)
  plt.plot(item3, reward3, color='mediumorchid', label='DDPG-3L-6M',alpha=0.9)
  plt.plot(item2, reward2, color='darkgreen', label='Feature-3L-sub5-6M',alpha=0.9)
  plt.plot(item1, reward1, color='tomato', label='PER-3L-6M',alpha=0.9)
  
  plt.legend()
  plt.xlabel('episode')
  plt.ylabel('reward')
  # plt.xlabel('step')
  # plt.ylabel('loss')
  plt.show()
  
path1='./helicopter_logs/PER-3L-6M/events.out.tfevents.1648093014.MacBook-Pro-3.local'
path2='./helicopter_logs/Feature-3L-sub5-6M/events.out.tfevents.1648104928.MacBook-Pro-3.local'
path3='./helicopter_logs/DDPG-3L-6M/events.out.tfevents.1648103688.MacBook-Pro-3.local'

path4='./helicopter_logs/ablation_experiment/DDPG-3L-6M-sub5/events.out.tfevents.1648102459.MacBook-Pro-3.local'
path5='./helicopter_logs/ablation_experiment/Feature-3L-6M/events.out.tfevents.1648102430.MacBook-Pro-3.local'
path6='./helicopter_logs/ablation_experiment/Feature-3L-sub5-3M/events.out.tfevents.1648093676.MacBook-Pro-3.local'
# old_data=getMemorySize(path1)
# new_item1,new_reward1=styleMemorySize(old_data)

# old_data=getMemorySize(path2)
# new_item2,new_reward2=styleMemorySize(old_data)

# old_data=getMemorySize(path3)
# new_item3,new_reward3=styleMemorySize(old_data)
# pltMemorySizeFig(new_item1,new_item2,new_item3,new_reward1,new_reward2,new_reward3)

old_data=getLoss(path1,'actor_loss')
new_item1,new_reward1=styleMemorySize(old_data)

old_data=getLoss(path2,'actor_loss')
new_item2,new_reward2=styleMemorySize(old_data)

old_data=getLoss(path3,'actor_loss')
new_item3,new_reward3=styleMemorySize(old_data)
pltLossFig(new_item1,new_item2,new_item3,new_reward1,new_reward2,new_reward3,'Actor Loss')

old_data=getLoss(path1,'critic_loss')
new_item1,new_reward1=styleLoss(old_data)

old_data=getLoss(path2,'critic_loss')
new_item2,new_reward2=styleLoss(old_data)

old_data=getLoss(path3,'critic_loss')
new_item3,new_reward3=styleLoss(old_data)
pltLossFig(new_item1,new_item2,new_item3,new_reward1,new_reward2,new_reward3,'Critic Loss')

old_data=getLoss(path1,'long_pos_reward_in_eipsode')
new_item1,new_reward1=styleLoss(old_data)

old_data=getLoss(path2,'long_pos_reward_in_eipsode')
new_item2,new_reward2=styleLoss(old_data)

old_data=getLoss(path3,'long_pos_reward_in_eipsode')
new_item3,new_reward3=styleLoss(old_data)
pltLossFig(new_item1,new_item2,new_item3,new_reward1,new_reward2,new_reward3,'Rewards in episode')