from DDPGModel import *
import gym
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import time
import random
def reset_reward(x):
  if x>=0.5:
    return 10
  reward=x+0.5
  if x>=-0.5:
    if x<=-0.2:
      reward =x+0.5
    elif x<-0.15:
      reward =x+0.5+0.2
    elif x<=0.1:
      reward =x+0.5+0.5
    else:
      reward=x+0.5+0.7
  else:
    reward=0
  return reward  
if __name__ == '__main__':
    ct = time.time()
    # env = gym.make('Pendulum-v1')
    env = gym.make('MountainCarContinuous-v0')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_bound = [-env.action_space.high[0], env.action_space.high[0]]

    ddpg = DDPG(obs_dim, act_dim, act_bound,capacity=10000)

    MAX_EPISODE = 100
    MAX_STEP = 2000
    update_every = 50
    batch_size = 128
    rewardList = []
    write=SummaryWriter('./gym/per')
    learn_num=0
    for episode in range(MAX_EPISODE):
      o = env.reset()
      ep_reward = 0
     
      
      step_num=0
      # while True:
      for j in range(MAX_STEP):
        # env.render()
        a = ddpg.get_action(o)
        
            
        o2, r, d, _ = env.step(a)
        r=reset_reward(o2[0])
        ddpg.store((o, a, r, o2, d))

        # if episode >= 5 and j % update_every == 0:
        #     for _ in range(update_every):
                # ddpg.update(batch_size=batch_size)
        learn_num+=ddpg.update(batch_size=batch_size)
        o = o2
        ep_reward += r
        step_num+=1
        if d:
          break
      ddpg.noise_decay()
      print('Episode:', episode, 'Reward: ' ,ep_reward, 'Step: ' ,step_num)
      write.add_scalar('learn_step_reward',ep_reward,learn_num)
      write.add_scalar('reward',ep_reward,episode)
      write.add_scalar('step',step_num,episode)
      write.add_scalar('every_reward',ep_reward/step_num,episode)
      write.add_scalar('memory',ddpg.sumTree.data.size,episode)
      rewardList.append(ep_reward)
    print('per  time  :',time.time()-ct)
    plt.figure()
    plt.plot(np.arange(len(rewardList)),rewardList)
    plt.show()
