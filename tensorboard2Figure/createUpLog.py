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
  long_pos_reward=ea.scalars.Items('reward')
  print(len(long_pos_reward))
  return long_pos_reward

def pltFig(item,reward1,reward2,reward3):
  
  plt.title('Rewards in eipsode')
  plt.plot(item, reward3, color='mediumorchid', label='Xplane-up-10000')
  plt.plot(item, reward2, color='darkorange', label='Xplane-up-5000', alpha=0.7)
  plt.plot(item, reward1, color='darkcyan', label='Xplane-up-3000',alpha=0.7)
  
  plt.legend()
  plt.xlabel('eipsode')
  plt.ylabel('reward')
  plt.show()

def style(data):
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
    
path1='./up/Xplane-up-3000/events.out.tfevents.1646812289.LAPTOP-GA832Q29'
path2='./up/Xplane-up-5000/events.out.tfevents.1646812303.LAPTOP-GA832Q29'
path3='./up/Xplane-up-10000/events.out.tfevents.1646812318.LAPTOP-GA832Q29'

old_data=getData(path1)
new_item1,new_reward1=style(old_data)


old_data=getData(path2)
new_item2,new_reward2=style(old_data)

old_data=getData(path3)
new_item3,new_reward3=style(old_data)
pltFig(new_item1,new_reward1,new_reward2,new_reward3)
