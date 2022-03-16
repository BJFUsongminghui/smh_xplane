from DDPGModel import *
import gym
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import time
if __name__ == '__main__':
    ct = time.time()
    # env = gym.make('Pendulum-v1')
    env = gym.make('MountainCarContinuous-v0')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_bound = [-env.action_space.high[0], env.action_space.high[0]]

    ddpg = DDPG(obs_dim, act_dim, act_bound,capacity=10000)

    MAX_EPISODE = 100
    MAX_STEP = 500
    update_every = 50
    batch_size = 128
    rewardList = []
    write=SummaryWriter('./gym/per')
    learn_num=0
    for episode in range(MAX_EPISODE):
      o = env.reset()
      ep_reward = 0
      for j in range(MAX_STEP):
        if episode > 5:
            a = ddpg.get_action(o, ddpg.act_noise)
        else:
            a = env.action_space.sample()
        o2, r, d, _ = env.step(a)
        ddpg.store((o, a, r, o2, d))

        # if episode >= 5 and j % update_every == 0:
        #     for _ in range(update_every):
                # ddpg.update(batch_size=batch_size)
        learn_num+=ddpg.update(batch_size=batch_size)
        o = o2
        ep_reward += r

        if d:
            break
      print('Episode:', episode, 'Reward:%i' % ep_reward)
      write.add_scalar('learn_step_reward',ep_reward,learn_num)
      write.add_scalar('reward',ep_reward,episode)
      rewardList.append(ep_reward)
    print('per  time  :',time.time()-ct)
    plt.figure()
    plt.plot(np.arange(len(rewardList)),rewardList)
    plt.show()
