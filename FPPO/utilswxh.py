import numpy as np


def aggregate_models(agent_models, Rs, global_models) -> list:
    if global_models is None:
        global_models = [agent_models[0].actor_net.state_dict(), agent_models[0].critic_net.state_dict()]

    for key in global_models[0]:
        global_models[0][key] -= global_models[0][key]
        for i, agent_model in enumerate(agent_models):
            global_models[0][key] += (Rs[i]/sum(Rs)) * agent_model.actor_net.state_dict()[key]

    for key in global_models[1]:
        global_models[1][key] -= global_models[1][key]
        for i, agent_model in enumerate(agent_models):
            global_models[1][key] += (Rs[i]/sum(Rs)) * agent_model.critic_net.state_dict()[key]

    return global_models


# 只聚合模型的隐藏层
# aggregate_mode: 1: r weighter 2: average 3: shop size 4: object 5: obj * shop size  6: select models
def aggregate_hidden_models(agent_models, Rs, global_models, aggregate_mode) -> list:
    assert aggregate_mode in [1, 2, 3, 4, 5,   6, 7]
    aggregate_keys_actor = ['fc2.weight', 'fc2.bias']
    aggregate_keys_critic = ['fc2.weight', 'fc2.bias', 'state_value.weight', 'state_value.bias']

    if global_models is None or global_models == 'None':  # 这里其实就是模型初始化的地方
        global_models = [{eve: agent_models[0].actor_net.state_dict()[eve] for eve in aggregate_keys_actor},
                         {eve: agent_models[0].critic_net.state_dict()[eve] for eve in aggregate_keys_critic}]

    aggregate_weights = []
    agent_num = len(agent_models)
    select_top_best = 4  # 选择前几个模型进行聚合
    # print(aggregate_mode)
    if aggregate_mode == 6:
        # 按照reward挑选前几个models
        sorted_rs = np.argsort(-np.array(Rs)).tolist()
        agent_models_sort = [agent_models[sorted_rs[i]] for i in range(select_top_best)]
        agent_id_sort = sorted_rs[:select_top_best]
        print('selected agent ids: ', agent_id_sort)
    else:
        agent_id_sort = [_ for _ in range(agent_num)]

    for agent_i in agent_id_sort:
        if aggregate_mode == 1:
            aggregate_weights.append(Rs[agent_i])
        elif aggregate_mode == 2:
            aggregate_weights.append(1)
        elif aggregate_mode == 3:
            aggregate_weights.append(agent_models[agent_i].actor_net.state_dict()['fc1.weight'].shape[1])
        elif aggregate_mode == 4:
            aggregate_weights.append(1 / Rs[agent_i])  # 此时的Rs传入的是obj
        elif aggregate_mode == 5:
            aggregate_weights.append(agent_models[agent_i].actor_net.state_dict()['fc1.weight'].shape[1] / Rs[agent_i]) # 此时的Rs传入的是obj
        elif aggregate_mode == 6:
            aggregate_weights.append(Rs[agent_i])
        elif aggregate_mode == 7:
            aggregate_weights.append(agent_models[agent_i].actor_net.state_dict()['fc1.weight'].shape[1] / Rs[agent_i])

    fenmu = sum(aggregate_weights)
    if aggregate_mode == 6:
        for key in global_models[0]:
            global_models[0][key] -= global_models[0][key]
            for i, agent_model in enumerate(agent_models_sort):
                global_models[0][key] += (aggregate_weights[i] / fenmu) * agent_model.actor_net.state_dict()[key]

        for key in global_models[1]:
            global_models[1][key] -= global_models[1][key]
            for i, agent_model in enumerate(agent_models_sort):
                global_models[1][key] += (aggregate_weights[i] / fenmu) * agent_model.critic_net.state_dict()[key]
    elif aggregate_mode == 7:
        assert len(aggregate_weights) == 6
        # 挑选weights最大的前4个聚合
        weight_sort = np.argsort(-np.argsort(aggregate_weights)).tolist()
        agent_models_sort = [agent_models[weight_sort[i]] for i in range(select_top_best)]
        print('selected agent ids: ', weight_sort[:select_top_best], 'total agents:', weight_sort)

        for key in global_models[0]:
            global_models[0][key] -= global_models[0][key]
            for i, agent_model in enumerate(agent_models_sort):
                global_models[0][key] += (aggregate_weights[i] / fenmu) * agent_model.actor_net.state_dict()[key]

        for key in global_models[1]:
            global_models[1][key] -= global_models[1][key]
            for i, agent_model in enumerate(agent_models_sort):
                global_models[1][key] += (aggregate_weights[i] / fenmu) * agent_model.critic_net.state_dict()[key]

    else:
        for key in global_models[0]:
            global_models[0][key] -= global_models[0][key]
            for i, agent_model in enumerate(agent_models):
                global_models[0][key] += (aggregate_weights[i]/fenmu) * agent_model.actor_net.state_dict()[key]

        for key in global_models[1]:
            global_models[1][key] -= global_models[1][key]
            for i, agent_model in enumerate(agent_models):
                global_models[1][key] += (aggregate_weights[i]/fenmu) * agent_model.critic_net.state_dict()[key]

    return global_models


# 只聚合critic模型的隐藏层+输出层
def aggregate_critic_models(agent_models, Rs, global_models, aggregate_mode) -> list:
    assert aggregate_mode in [1, 2, 3]
    aggregate_keys = ['fc2.weight', 'fc2.bias', 'state_value.weight', 'state_value.bias']

    if global_models is None or global_models == 'None':  # 这里其实就是模型初始化的地方
        global_models = [{},
                         {eve: agent_models[0].critic_net.state_dict()[eve] for eve in aggregate_keys}]

    aggregate_weights = []
    agent_num = len(agent_models)
    for agent_i in range(agent_num):
        if aggregate_mode == 1:
            aggregate_weights.append(Rs[agent_i])
        elif aggregate_mode == 2:
            aggregate_weights.append(1)
        elif aggregate_mode == 3:
            aggregate_weights.append(agent_models[agent_i].actor_net.state_dict()['fc1.weight'].shape[1])

    fenmu = sum(aggregate_weights)

    for key in global_models[1]:
        global_models[1][key] -= global_models[1][key]
        for i, agent_model in enumerate(agent_models):
            global_models[1][key] += (aggregate_weights[i]/fenmu) * agent_model.critic_net.state_dict()[key]

    return global_models


# mp
def func(async_args):
    # print(async_args)
    assert len(async_args) == 4
    An_agent_training_episode_for_an_Fed_inter = async_args[0]
    index = async_args[1][0]
    alg_name = async_args[1][1]
    eve_model = async_args[2]
    global_models = async_args[3]
    min_obj, epi, R, Obj = eve_model.train(An_agent_training_episode_for_an_Fed_inter, index, global_models, alg_name)

    return [min_obj, epi, R, Obj]


# single ppo, ac, sac
def func_single_pgs(as_args):
    assert len(as_args) == 4

    each_file_list = as_args[0]
    JssEnv = as_args[1]
    PPO = as_args[2]
    epoch = as_args[3]

    envs = []
    for each_file in each_file_list:
        env = JssEnv(env_config={'instance_path': each_file, 'dynamic_rate': 0.0,
                                 'alg_name': PPO.alg})
        envs.append(env)
    # env = JssEnv(fixed_env_config)
    scale = envs[0].jobs * envs[0].machines
    model = PPO(envs, batch_size=4 * scale, clip_ep=0.541)
    a, b, c = model.train(epoch)

    return [each_file_list[0].split('/')[-1], a, b, c]


# single dqn, ddqn, dueling dqn
def func_fingle_dqns(as_args):
    each_file_list = as_args[0]
    JssEnv = as_args[1]
    args = as_args[2]
    Agent = as_args[3]
    Epoch = as_args[4]

    envs = []
    for each_file in each_file_list:
        env = JssEnv(env_config={'instance_path': each_file, 'alg_name': Agent.alg,
                                 'dynamic_rate': 0.0,})
        envs.append(env)

    args['envs'] = envs
    agent = Agent(**args)
    a, b = agent.train(Epoch)

    return [each_file_list[0].split('/')[-1], a, b]


def copy_model_over(from_model, to_model):
    """Copies model parameters from from_model to to_model"""
    for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
        to_model.data.copy_(from_model.data.clone())