
import torch
from JSSEnv.envs.jss_env import JssEnv
from FPPO.SinglePPO import PPO

from FPPO.utilswxh import aggregate_hidden_models, aggregate_critic_models, func
import os
import torch.multiprocessing as multiprocessing
import argparse
torch.set_num_threads(1)
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo')
    print('可用的CPU数量：', os.cpu_count())

    parser.add_argument('-alg', type=str, default='sph5', help='h1 h2 h5 sph1 sph5')
    args = parser.parse_args()

    n_agents = 6  # 6 对应了6个边缘端
    Fed_iterations = 10  # 共进行几次聚合 10
    An_agent_training_episode_for_an_Fed_inter = 500  # 每次聚合间隔每个边缘端训练多少代 200
    alg_name = args.alg  # h1 h2 h3 c1 c2 c3 ippo

    print('--An agent trains for {} episodes in total--'.format(Fed_iterations *
                                                                An_agent_training_episode_for_an_Fed_inter))

    case_root = './data/'
    paths = [case_root + eve for eve in ['edge1/train/', 'edge2/train/', 'edge3/train/', 'edge4/train/', 'edge5/train/',
                                         'edge6/train/']]

    assert len(paths) >= n_agents, 'The files are not enough for edge agents !!!'
    assert n_agents <= os.cpu_count(), 'Be careful of your computer !!!'
    assert alg_name != 'ippo', 'This main function does not implement this one !!!!!!!!'

    # 实例化n_agents个边缘环境
    agent_models = []
    for a in range(n_agents):

        envs = [JssEnv({'instance_path': paths[a] + train_case_name, 'dynamic_rate': 0.0, 'alg_name': alg_name})
                for train_case_name in os.listdir(paths[a])]

        scale = envs[0].jobs * envs[0].machines
        # model的形状只与job的数量有关系，而与机器的数量没关系
        model = PPO(envs, batch_size=4*scale, clip_ep=0.541)
        agent_models.append(model)

    # 记录一些结果
    min_objs = [[paths[_]] for _ in range(n_agents)]
    # 开始联邦训练，先弄顺序执行的看看大致效果，后面再并行化处理
    # 采用多进程模拟多个边缘端工厂的分布式执行过程
    global_models = None
    for iter in range(Fed_iterations):
        args_list = [[An_agent_training_episode_for_an_Fed_inter,
                      [each_index, alg_name],  # 这个当作一个传参列表了
                      agent_models[each_index],
                      global_models] for each_index in range(n_agents)]
        # 从这里并发执行
        pool = multiprocessing.Pool(n_agents)
        results = pool.map_async(func, args_list)
        Rs = []
        a = 0
        Objs = []
        for res in results.get():
            Rs.append(res[2])
            Objs.append(res[3])
            min_objs[a].append(res[0])
            a += 1
        pool.close()
        pool.join()
        # 开始聚合 这里有多种聚合方式可以选择
        # global_models = aggregate_models(agent_models, Rs, global_models)  # 最原始 a c 所有参数 基本上用不上
        # -------------------------------------------------------------------------------------------------
        # print(alg_name, alg_name == 'h1', alg_name == 'h1' or '156456')
        if (alg_name == 'h1') or (alg_name == 'ph1'):
            global_models = aggregate_hidden_models(agent_models, Rs, global_models, 1)  # a c 隐藏层+c output层
        elif (alg_name == 'h2') or (alg_name == 'ph2'):
            global_models = aggregate_hidden_models(agent_models, Rs, global_models, 2)  # a c 隐藏层+c output层
        elif (alg_name == 'h3') or (alg_name == 'ph3'):
            global_models = aggregate_hidden_models(agent_models, Rs, global_models, 3)  # a c 隐藏层+c output层
        elif (alg_name == 'h4') or (alg_name == 'ph4'):
            global_models = aggregate_hidden_models(agent_models, Objs, global_models, 4)  # a c 隐藏层+c output层
        elif (alg_name == 'h5') or (alg_name == 'ph5'):
            global_models = aggregate_hidden_models(agent_models, Objs, global_models, 5)  # a c 隐藏层+c output层
        elif alg_name == 'sph1':
            global_models = aggregate_hidden_models(agent_models, Objs, global_models, 6)  # a c 隐藏层+c output层
        elif alg_name == 'sph5':
            global_models = aggregate_hidden_models(agent_models, Objs, global_models, 7)  # a c 隐藏层+c output层

        elif alg_name == 'c1':
            global_models = aggregate_critic_models(agent_models, Rs, global_models, 1)  # critic 隐藏层+output层
        elif alg_name == 'c2':
            global_models = aggregate_critic_models(agent_models, Rs, global_models, 2)  # critic 隐藏层+output层
        elif alg_name == 'c3':
            global_models = aggregate_critic_models(agent_models, Rs, global_models, 3)  # critic 隐藏层+output层

    # 开始评估训练的结果
    print('=============================================begin test ====================================================')
    test_cases_roots = ['./data/edge1/test/', './data/edge2/test/', './data/edge3/test/', './data/edge4/test/',
                        './data/edge5/test/', './data/edge6/test/', ]

    for i, each_train_instance_name in enumerate(['ta0', 'ta1', 'ta2', 'ta3', 'ta4', 'ta5']):
        test_root = test_cases_roots[i]
        test_case_names = os.listdir(test_root)
        envs = [JssEnv({'instance_path': test_root + each_test_instance_name, 'dynamic_rate': 0.01, 'alg_name': alg_name})
                for each_test_instance_name in test_case_names]
        model = PPO(envs, batch_size=1, clip_ep=0.541)
        model.test(each_train_instance_name, alg_name)
