import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import copy


class Actor(nn.Module):
    def __init__(self, num_input, num_output, node_num=400):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_input, node_num)
        self.fc2 = nn.Linear(node_num, node_num)
        self.fc3 = nn.Linear(node_num, 100)  # 多加一层
        self.action_head = nn.Linear(100, num_output)

    def forward(self, x, mask):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.action_head(x)
        x[mask == False] = -torch.inf
        action_prob = F.softmax(x, dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, num_input, num_output=1, node_num=400):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_input, node_num)
        self.fc2 = nn.Linear(node_num, node_num)
        self.fc3 = nn.Linear(node_num, 100)  # 多加一层
        self.state_value = nn.Linear(100, num_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.state_value(x)
        return value


class PPO:
    def __init__(self, j_envs, unit_num=400, batch_size=32, clip_ep=0.2):
        super(PPO, self).__init__()

        self.envs = j_envs
        self.batch_size = batch_size  # update batch size
        self.epsilon = clip_ep

        self.state_dim = self.envs[0].jobs * 7
        self.action_dim = self.envs[0].jobs + 1
        self.case_name = self.envs[0].case_name
        self.gamma = 1  # reward discount
        self.A_LR = 6e-4  # learning rate for actor
        self.C_LR = 3e-3  # learning rate for critic
        self.A_UPDATE_STEPS = 16  # actor update steps
        self.max_grad_norm = 0.5
        self.training_step = 0

        self.actor_net = Actor(self.state_dim, self.action_dim, node_num=1*unit_num)
        self.critic_net = Critic(self.state_dim, node_num=unit_num)
        self.actor_optimizer = optimizer.Adam(self.actor_net.parameters(), self.A_LR)
        self.actor_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.actor_optimizer, [500, 2500, 5000], gamma=0.5, last_epoch=-1)

        self.critic_net_optimizer = optimizer.Adam(self.critic_net.parameters(), self.C_LR)
        self.critic_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.critic_net_optimizer, [500, 2500, 5000], gamma=0.5, last_epoch=-1)

        if not os.path.exists('param'):
            os.makedirs('param/net_param')

        # ---------------------------------------------------------------------------------------------------
        self.is_prox = False  # 不采用近端优化策略
        self.is_klp = False  # actor用pi约束 + critic用prox约束
        self.mu = 1e-3  # 这个参数决定的约束的大小
        self.global_critic_net_t = Critic(self.state_dim, node_num=unit_num)
        self.global_actor_net_t = Actor(self.state_dim, self.action_dim, node_num=1 * unit_num)

    # select with mask that has the same dimension with action_prob
    def select_action(self, state, mask):
        state = torch.from_numpy(state).float().unsqueeze(0)
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state, mask)
        # try:
        c = Categorical(action_prob)
        # except:
        #     print(action_prob, self.actor_net(state))
        #     # for eve in self.actor_net.parameters():
        #     #     print(eve)
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def get_value(self, state):
        state = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_params(self):
        torch.save(self.actor_net.state_dict(), './models/' + self.envs[0].shop_name[:-1] + self.envs[0].alg_name + 'actor_net.model')
        torch.save(self.critic_net.state_dict(), './models/' + self.envs[0].shop_name[:-1] + self.envs[0].alg_name + 'critic_net.model')

    def save_params_with_performance(self, performance):
        torch.save(self.actor_net.state_dict(), './models/' + self.envs[0].shop_name[:-1] + self.envs[0].alg_name + 'actor_net-{}.model'.format(performance))
        # torch.save(self.critic_net.state_dict(), './models/' + self.envs[0].shop_name[:-1] + self.envs[0].alg_name + 'critic_net.model')

    def load_params(self, train_shop_name, selected_alg_name, mode=None):
        if mode == 'test':
            self.actor_net.load_state_dict(
                torch.load('./models/' + train_shop_name + selected_alg_name + 'actor_net.model'))
        else:
            self.actor_net.load_state_dict(torch.load('./models/' + train_shop_name + selected_alg_name + 'actor_net.model'))
            self.critic_net.load_state_dict(torch.load('./models/' + train_shop_name + selected_alg_name + 'critic_net.model'))

    # 这个load的策略是有啥load啥
    def load_model_from_mem(self, models):
        # try:
            self.critic_net.load_state_dict(models[1], strict=False)
            self.actor_net.load_state_dict(models[0], strict=False)
        #     pass
        # except RuntimeError:  # 否则只load隐藏层
        #     for each_key in models[1]:
        #         print('待Load的模型：', models[1][each_key])
        #         print('load 前：', self.critic_net.state_dict()[each_key])
        #         self.critic_net.state_dict()[each_key] = copy.deepcopy(models[1][each_key])
        #         print('load 后：', self.critic_net.state_dict()[each_key])
        #     for each_key in models[0]:
        #         self.actor_net.state_dict()[each_key] = copy.deepcopy(models[0][each_key])

    def load_prox_global_ac_from_mem(self, models):
        self.global_actor_net_t.load_state_dict(models[0], strict=False)
        self.global_critic_net_t.load_state_dict(models[1], strict=False)

    def update(self, bs, ba, br, bp, bm):
        # get old actor log prob
        old_action_log_prob = torch.tensor(bp, dtype=torch.float).view(-1, 1)
        state = torch.tensor(np.array(bs), dtype=torch.float)
        mask = torch.tensor(np.array(bm), dtype=torch.float)
        action = torch.tensor(ba, dtype=torch.long).view(-1, 1)
        d_reward = torch.tensor(br, dtype=torch.float)
        # 打印衰减的学习率
        # print(self.actor_optimizer.state_dict()['param_groups'][0]['lr'])
        # print(self.critic_net_optimizer.state_dict()['param_groups'][0]['lr'])

        for i in range(self.A_UPDATE_STEPS):
            for index in BatchSampler(SubsetRandomSampler(range(len(ba))), self.batch_size, True):
                #  compute the advantage
                d_reward_index = d_reward[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = d_reward_index - V
                advantage = delta.detach()

                # epoch iteration, PPO core!
                # action_prob == nan
                action_prob = self.actor_net(state[index], mask[index]).gather(1, action[index])  # new policy
                ratio = (action_prob / old_action_log_prob[index])
                surrogate = ratio * advantage
                clip_loss = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
                if self.is_prox:
                    proxi_term = 0.0
                    # print(self.actor_net.parameters())
                    # for w in self.actor_net.parameters():
                    #     print(w.shape)
                    # for w in self.critic_net.parameters():
                    #     print(w.shape)

                    layer_i = -1  # 因为paras是generator
                    for w, w_t in zip(self.actor_net.parameters(), self.global_actor_net_t.parameters()):
                        layer_i += 1
                        if layer_i in [0, 6, 7]: # 这个东西和网络的深度有关
                            proxi_term += 0
                        else:
                            proxi_term += (w - w_t).norm(2)
                    assert layer_i == 7, 'nn layers change !!!'

                    # for w, w_t in zip(self.actor_net.parameters(), self.global_model[0].parameters()):
                    #     proxi_term += (w - w_t).norm(2)
                    action_loss = -torch.min(surrogate, clip_loss).mean() + (self.mu / 2) * proxi_term
                    # print((self.mu / 2) * proxi_term)
                    # print(action_loss, 666666666666666)
                else:
                    if self.is_klp:
                        action_loss = -torch.min(surrogate, clip_loss).mean() + \
                                      1e2 * self.mu * F.kl_div(torch.log(action_prob), old_action_log_prob[index])
                        # print(self.mu * F.kl_div(torch.log(action_prob), old_action_log_prob[index]))
                        # print(action_loss, 7777777777777777)

                    else:
                        action_loss = -torch.min(surrogate, clip_loss).mean()
                # for (eve, para) in self.actor_net.parameters():
                #     if torch.isnan(para):
                #         print(eve)
                if torch.isnan(action_loss):
                    print(action_prob.view(-1))
                    assert False

                # update actor network
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.actor_scheduler.step()

                # update critic network
                if self.is_prox:
                    proxi_term2 = 0.0
                    # for w in self.critic_net.parameters():
                    #     print(w.shape)
                    layer_i2 = -1  # 因为paras是generator
                    for w2, w_t2 in zip(self.critic_net.parameters(), self.global_critic_net_t.parameters()):
                        layer_i2 += 1
                        if layer_i2 in [0]:  # 这个东西和网络的深度有关
                            proxi_term2 += 0
                        else:
                            proxi_term2 += (w2 - w_t2).norm(2)
                    assert layer_i2 == 7, 'nn layers change !!!'
                    # prox 的 loss
                    value_loss = F.mse_loss(d_reward_index, V) + (self.mu / 2) * proxi_term2
                else:
                    value_loss = F.mse_loss(d_reward_index, V)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.critic_scheduler.step()

                self.training_step += 1
                # print(action_loss, value_loss)

    def test(self, shop_name, alg_name):
        self.load_params(shop_name, alg_name, 'test')
        value = []
        for m in range(10):
            for each_env in self.envs:
                allstate = each_env.reset()
                state = allstate['real_obs'].flatten()
                mask = allstate['action_mask']
                for _ in range(5000):
                    action, _ = self.select_action(state, mask)
                    next_state, reward, done, _ = each_env.step(action, m)
                    mask = next_state['action_mask']
                    next_state = next_state['real_obs'].flatten()
                    state = next_state
                    if done:
                        break
                value.append(each_env.current_time_step)
        return min(value)

    def train(self, epoches, agenti, global_models=None, alg_name=None):
        if global_models is not None:
            assert len(global_models) == 2, 'an actor and a critic'
            self.load_model_from_mem(global_models)
            if alg_name in ['ph1', 'ph2', 'ph3', 'ph4', 'ph5']:
                self.is_prox = True
                self.load_prox_global_ac_from_mem(global_models)  # 更新t时刻的ac全局模型
            if alg_name == 'kh5':
                self.load_prox_global_ac_from_mem(global_models)  # 更新t时刻的ac全局模型
                self.is_klp = True

        index = 0
        converged = 0
        converged_value = []
        episode_rewards = []
        objs = []
        # t0 = time.time()
        for i_epoch in range(epoches):
            # if time.time()-t0 >= 3600:
            #     break
            bs, ba, br, bp, bm = [], [], [], [], []
            # for m in range(self.memory_size):  # memory size is the number of complete episode
            for env_i, each_env in enumerate(self.envs):
                buffer_s, buffer_a, buffer_r, buffer_p, buffer_m = [], [], [], [], []
                all_state = each_env.reset()
                state = all_state['real_obs'].flatten()
                action_mask = all_state['action_mask']
                episode_reward = 0
                for _ in range(5000):
                    action, action_prob = self.select_action(state, action_mask)
                    buffer_m.append(copy.deepcopy(action_mask))
                    next_state, reward, done, _ = each_env.step(action, i_epoch)
                    # print(action_mask, done)

                    action_mask = next_state['action_mask']
                    next_state = next_state['real_obs'].flatten()
                    buffer_s.append(state)
                    buffer_a.append(action)
                    buffer_r.append(reward)
                    buffer_p.append(action_prob)

                    state = next_state
                    episode_reward += reward
                    if done:
                        v_s_ = 0
                        discounted_r = []
                        for r in buffer_r[::-1]:
                            v_s_ = r + self.gamma * v_s_
                            discounted_r.append(v_s_)
                        discounted_r.reverse()
                        episode_rewards.append(episode_reward)
                        objs.append(-reward)

                        bs[len(bs):len(bs)] = buffer_s
                        bm[len(bm):len(bm)] = buffer_m
                        ba[len(ba):len(ba)] = buffer_a
                        br[len(br):len(br)] = discounted_r
                        bp[len(bp):len(bp)] = buffer_p
                        # Episode: make_span: Episode reward
                        print('agent {} Training ep: {}   current makespan: {}  r: {:.2f} ac_lr: {}'.format(agenti,
                                                        i_epoch,
                                                        each_env.current_time_step,
                                                        episode_reward,
                                                        self.actor_optimizer.state_dict()['param_groups'][0]['lr']))

                        index = i_epoch * len(self.envs) + env_i
                        converged_value.append(each_env.current_time_step)
                        if len(converged_value) >= 31:
                            converged_value.pop(0)
                        break
            self.update(bs, ba, br, bp, bm)
            converged = index
            # if ((max(converged_value)-min(converged_value)) < 5) and len(converged_value) >= 30:
            #     converged = index
            #     break
        if not os.path.exists('results'):
            os.makedirs('results')
        self.save_params()
        self.save_params_with_performance(sum(episode_rewards) / len(episode_rewards))
        return min(converged_value), converged, sum(episode_rewards) / len(episode_rewards), sum(objs)/len(objs)
