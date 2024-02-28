import bisect
import copy
import datetime
import random
import time

import pandas as pd
import gym
import numpy as np
import plotly.figure_factory as ff
from pathlib import Path

"""
Try to balance whether the eighth state is necessary in the studied problem
"""

class JssEnv(gym.Env):
    def __init__(self, env_config=None):
        """
        This environment model the job shop scheduling problem as a single agent problem:

        -The actions correspond to a job allocation + one action for no allocation at
        this time step (NOPE action)

        -We keep a time with next possible time steps

        -Each time we allocate a job, the end of the job is added to the stack of
        time steps

        -If we don't have a legal action (i.e. we can't allocate a job),
        we automatically go to the next time step until we have a legal action

        -
        :param env_config: Ray dictionary of config parameter
        """
        # if env_config is None:
        #     env_config = {
        #         "instance_path": str(Path(__file__).parent.absolute())
        #         + "/instances/ta80"
        #     }
        instance_path = env_config["instance_path"]

        self.alg_name = env_config['alg_name']

        # 记录环境的一些变化
        self.record_path = './logs/'+instance_path.split('/')[-1] + '-' + self.alg_name
        self.env_call_num = 0
        with open(self.record_path, 'w') as f3:
            pass

        # 读取random估计的值
        with open('./data/value', 'r') as f0:
            a = [eve.strip() for eve in f0.readlines()]
            estimate_dict = {}
            for line in a:
                if line != '':
                    temp = line.split()
                    estimate_dict[temp[0]] = [float(ev) for ev in temp[1:]]
        self.max_t = estimate_dict[instance_path][0]
        self.max_c = estimate_dict[instance_path][1]

        # 获取动态性的参数,processing time需要动态变化
        try:
            self.dynamic_rate = env_config['dynamic_rate']
        except KeyError:
            self.dynamic_rate = 0.00  # No dynamics
            print('The system dynamic is off !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n !!!!!!!!!!!!!!!!! \n!!!!!!!!!!!!!!!!')

        self.case_name = instance_path

        self.shop_name = instance_path.split('/')[-1]

        self.dynamic_changed_jobs = None
        self.c = None

        # initial values for variables used for instance
        self.jobs = 0
        self.machines = 0
        self.instance_matrix = None
        self.jobs_length = None
        self.max_time_op = 0
        self.max_time_jobs = 0
        self.nb_legal_actions = 0
        self.nb_machine_legal = 0
        # initial values for variables used for solving (to reinitialize when reset() is called)
        self.solution = None
        self.last_solution = None
        self.last_time_step = float("inf")
        self.current_time_step = float("inf")
        self.next_time_step = list()
        self.next_jobs = list()
        self.legal_actions = None
        self.time_until_available_machine = None
        self.time_until_finish_current_op_jobs = None
        self.todo_time_step_job = None  # 每个job当前时刻之后需要完成的operations
        self.total_perform_op_time_jobs = None
        self.needed_machine_jobs = None
        self.total_idle_time_jobs = None
        self.idle_time_jobs_last_op = None
        self.state = None
        self.illegal_actions = None
        self.action_illegal_no_op = None
        self.machine_legal = None
        # initial values for variables used for representation
        self.start_timestamp = datetime.datetime.now().timestamp()
        self.sum_op = 0
        instance_file = open(instance_path, "r")
        line_str = instance_file.readline()
        line_cnt = 1
        while line_str:
            split_data = line_str.split()
            if line_cnt == 1:
                self.jobs, self.machines = int(split_data[0]), int(split_data[1])
                # matrix which store tuple of (machine, length of the job)
                self.instance_matrix = np.zeros(
                    (self.jobs, self.machines), dtype=(int, 2)
                )
                # contains all the time to complete jobs
                self.jobs_length = np.zeros(self.jobs, dtype=int)
            else:
                # couple (machine, time)
                assert len(split_data) % 2 == 0
                # each jobs must pass a number of operation equal to the number of machines
                assert len(split_data) / 2 == self.machines
                i = 0
                # we get the actual jobs
                job_nb = line_cnt - 2
                while i < len(split_data):
                    machine, time = int(split_data[i]), int(split_data[i + 1])
                    self.instance_matrix[job_nb][i // 2] = (machine, time)
                    self.max_time_op = max(self.max_time_op, time)
                    self.jobs_length[job_nb] += time
                    self.sum_op += time
                    i += 2
            line_str = instance_file.readline()
            line_cnt += 1
        instance_file.close()

        self.original_instance_matrix = copy.deepcopy(self.instance_matrix)

        with open('./data/paras', 'r') as f:
            a = [eve.strip() for eve in f.readlines()]
            res = []
            for line in a:
                if line != '':
                    res.append([float(eve) for eve in line.split()])
            assert len(res) == 3, 'new paras ???'
            self.T_warmup_machine = res[0]
            self.sigma_machine = res[1]
            self.u_k_p = res[2][0]
            self.u_k_w = res[2][1]

        self.max_time_jobs = max(self.jobs_length)
        # check the parsed data are correct
        assert self.max_time_op > 0
        assert self.max_time_jobs > 0
        assert self.jobs > 0
        assert self.machines > 1, "We need at least 2 machines"
        assert self.instance_matrix is not None
        # allocate a job + one to wait
        self.action_space = gym.spaces.Discrete(self.jobs + 1)
        # used for plotting
        self.colors = [
            tuple([random.random() for _ in range(3)]) for _ in range(self.machines)
        ]
        """
        matrix with the following attributes for each job:
            -Legal job
            -Left over time on the current op
            -Current operation %
            -Total left over time
            -When next machine available
            -Time since IDLE: 0 if not available, time otherwise
            -Total IDLE time in the schedule
        """
        self.observation_space = gym.spaces.Dict(
            {
                "action_mask": gym.spaces.Box(0, 1, shape=(self.jobs + 1,)),
                "real_obs": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(self.jobs, 7), dtype=float
                ),
            }
        )

    def _get_current_state_representation(self):
        self.state[:, 0] = self.legal_actions[:-1]
        return {
            "real_obs": self.state,
            "action_mask": self.legal_actions,
        }

    def get_legal_actions(self):
        return self.legal_actions

    def reset(self):
        self.episode_start_time = time.time()
        self.dynamic_changed_jobs = []
        self.hole_plannings = 0
        self.c = 0
        self.epi_r = []
        # print(1111,sum(sum(sum(self.instance_matrix))), sum(sum(sum(self.original_instance_matrix))))
        self.instance_matrix = copy.deepcopy(self.original_instance_matrix)
        # print(1111,sum(sum(sum(self.instance_matrix))), sum(sum(sum(self.original_instance_matrix))))

        self.current_time_step = 0
        self.next_time_step = list()
        self.next_jobs = list()
        self.nb_legal_actions = self.jobs
        self.nb_machine_legal = 0
        # represent all the legal actions
        self.legal_actions = np.ones(self.jobs + 1, dtype=bool)
        self.legal_actions[self.jobs] = False
        # used to represent the solution
        self.solution = np.full((self.jobs, self.machines), -1, dtype=int)
        self.time_until_available_machine = np.zeros(self.machines, dtype=int)
        self.time_until_finish_current_op_jobs = np.zeros(self.jobs, dtype=int)
        self.todo_time_step_job = np.zeros(self.jobs, dtype=int)
        self.total_perform_op_time_jobs = np.zeros(self.jobs, dtype=int)
        self.needed_machine_jobs = np.zeros(self.jobs, dtype=int)
        self.total_idle_time_jobs = np.zeros(self.jobs, dtype=int)
        self.idle_time_jobs_last_op = np.zeros(self.jobs, dtype=int)
        self.illegal_actions = np.zeros((self.machines, self.jobs), dtype=bool)
        self.action_illegal_no_op = np.zeros(self.jobs, dtype=bool)
        self.machine_legal = np.zeros(self.machines, dtype=bool)
        for job in range(self.jobs):
            needed_machine = self.instance_matrix[job][0][0]
            self.needed_machine_jobs[job] = needed_machine
            if not self.machine_legal[needed_machine]:
                self.machine_legal[needed_machine] = True
                self.nb_machine_legal += 1
        self.state = np.zeros((self.jobs, 7), dtype=float)
        return self._get_current_state_representation()

    def _prioritization_non_final(self):
        if self.nb_machine_legal >= 1:
            for machine in range(self.machines):
                if self.machine_legal[machine]:
                    final_job = list()
                    non_final_job = list()
                    min_non_final = float("inf")
                    for job in range(self.jobs):
                        if (
                            self.needed_machine_jobs[job] == machine
                            and self.legal_actions[job]
                        ):
                            if self.todo_time_step_job[job] == (self.machines - 1):
                                final_job.append(job)
                            else:
                                current_time_step_non_final = self.todo_time_step_job[
                                    job
                                ]
                                time_needed_legal = self.instance_matrix[job][
                                    current_time_step_non_final
                                ][1]
                                machine_needed_nextstep = self.instance_matrix[job][
                                    current_time_step_non_final + 1
                                ][0]
                                if (
                                    self.time_until_available_machine[
                                        machine_needed_nextstep
                                    ]
                                    == 0
                                ):
                                    min_non_final = min(
                                        min_non_final, time_needed_legal
                                    )
                                    non_final_job.append(job)
                    if len(non_final_job) > 0:
                        for job in final_job:
                            current_time_step_final = self.todo_time_step_job[job]
                            time_needed_legal = self.instance_matrix[job][
                                current_time_step_final
                            ][1]
                            if time_needed_legal > min_non_final:
                                self.legal_actions[job] = False
                                self.nb_legal_actions -= 1

    def _check_no_op(self):
        self.legal_actions[self.jobs] = False
        if (
            len(self.next_time_step) > 0
            and self.nb_machine_legal <= 3
            and self.nb_legal_actions <= 4
        ):
            machine_next = set()
            next_time_step = self.next_time_step[0]
            max_horizon = self.current_time_step
            max_horizon_machine = [
                self.current_time_step + self.max_time_op for _ in range(self.machines)
            ]
            for job in range(self.jobs):
                if self.legal_actions[job]:
                    time_step = self.todo_time_step_job[job]
                    machine_needed = self.instance_matrix[job][time_step][0]
                    time_needed = self.instance_matrix[job][time_step][1]
                    end_job = self.current_time_step + time_needed
                    if end_job < next_time_step:
                        return
                    max_horizon_machine[machine_needed] = min(
                        max_horizon_machine[machine_needed], end_job
                    )
                    max_horizon = max(max_horizon, max_horizon_machine[machine_needed])
            for job in range(self.jobs):
                if not self.legal_actions[job]:
                    if (
                        self.time_until_finish_current_op_jobs[job] > 0
                        and self.todo_time_step_job[job] + 1 < self.machines
                    ):
                        time_step = self.todo_time_step_job[job] + 1
                        time_needed = (
                            self.current_time_step
                            + self.time_until_finish_current_op_jobs[job]
                        )
                        while (
                            time_step < self.machines - 1 and max_horizon > time_needed
                        ):
                            machine_needed = self.instance_matrix[job][time_step][0]
                            if (
                                max_horizon_machine[machine_needed] > time_needed
                                and self.machine_legal[machine_needed]
                            ):
                                machine_next.add(machine_needed)
                                if len(machine_next) == self.nb_machine_legal:
                                    self.legal_actions[self.jobs] = True
                                    return
                            time_needed += self.instance_matrix[job][time_step][1]
                            time_step += 1
                    elif (
                        not self.action_illegal_no_op[job]
                        and self.todo_time_step_job[job] < self.machines  # 这个代表没加工完
                    ):
                        time_step = self.todo_time_step_job[job]
                        machine_needed = self.instance_matrix[job][time_step][0]
                        time_needed = (
                            self.current_time_step
                            + self.time_until_available_machine[machine_needed]
                        )
                        while (
                            time_step < self.machines - 1 and max_horizon > time_needed
                        ):
                            machine_needed = self.instance_matrix[job][time_step][0]
                            if (
                                max_horizon_machine[machine_needed] > time_needed
                                and self.machine_legal[machine_needed]
                            ):
                                machine_next.add(machine_needed)
                                if len(machine_next) == self.nb_machine_legal:
                                    self.legal_actions[self.jobs] = True
                                    return
                            time_needed += self.instance_matrix[job][time_step][1]
                            time_step += 1

    def step(self, action: int, epi=None):
        reward = 0.0
        # print(self.todo_time_step_job)
        if action == self.jobs:   # 没有动作可以执行时
            self.nb_machine_legal = 0
            self.nb_legal_actions = 0
            for job in range(self.jobs):
                if self.legal_actions[job]:
                    self.legal_actions[job] = False
                    needed_machine = self.needed_machine_jobs[job]
                    self.machine_legal[needed_machine] = False
                    self.illegal_actions[needed_machine][job] = True
                    self.action_illegal_no_op[job] = True
            while self.nb_machine_legal == 0:  # 一直减少到下一个合法动作出现
                reward -= self.increase_time_step()  # 奖励是负的
            scaled_reward = self._reward_scaler(reward)
            self._prioritization_non_final()
            self._check_no_op()
            self.epi_r.append(scaled_reward)
            return (
                self._get_current_state_representation(),
                scaled_reward,
                self._is_done(),
                {},
            )
        else:
            #  dynamicaly
            if random.randint(0,100) < self.dynamic_rate * 100:
                self.dynamic_processing_time()

            current_time_step_job = self.todo_time_step_job[action]
            machine_needed = self.needed_machine_jobs[action]
            time_needed = self.instance_matrix[action][current_time_step_job][1]  # p_{o_{i,j}}
            reward += time_needed
            self.time_until_available_machine[machine_needed] = time_needed
            self.time_until_finish_current_op_jobs[action] = time_needed
            self.state[action][1] = time_needed / self.max_time_op
            to_add_time_step = self.current_time_step + time_needed
            if to_add_time_step not in self.next_time_step:  # 把当前时刻插入到一些记录数组里
                index = bisect.bisect_left(self.next_time_step, to_add_time_step)
                self.next_time_step.insert(index, to_add_time_step)
                self.next_jobs.insert(index, action)
            self.solution[action][current_time_step_job] = self.current_time_step
            for job in range(self.jobs):  # 这里是做什么用的呢?
                if (
                    self.needed_machine_jobs[job] == machine_needed
                    and self.legal_actions[job]
                ):
                    self.legal_actions[job] = False
                    self.nb_legal_actions -= 1
            # 下面这两句是说把被选择的机器合法性去掉,表示这个机器已经被选择了
            self.nb_machine_legal -= 1
            self.machine_legal[machine_needed] = False
            for job in range(self.jobs):
                if self.illegal_actions[machine_needed][job]:
                    self.action_illegal_no_op[job] = False
                    self.illegal_actions[machine_needed][job] = False
            # if we can't allocate new job in the current timestep, we pass to the next one
            while self.nb_machine_legal == 0 and len(self.next_time_step) > 0:
                hole_planning = self.increase_time_step()
                reward -= hole_planning
                self.hole_plannings += hole_planning
            self._prioritization_non_final()
            self._check_no_op()
            # we then need to scale the reward
            scaled_reward = self._reward_scaler(reward)
            if self._is_done():

                cost = self.calculate_energy()
                scaled_reward = - (self.current_time_step /self.max_t + cost / self.max_c)

                epi_time = time.time() - self.episode_start_time
                # max_t和max_c用random_policy估计吧
                with open(self.record_path, 'a') as f3:
                    print('%-10s\t%-10s\t%-10s\t%-10s\t%-10s\t%-10s\t%-10s\t%-10s\t%-10s\t%-10s\t%-10s\t' % (
                                                        epi, len(self.epi_r), len(self.dynamic_changed_jobs),  # 0~2
                                                        round(sum(self.epi_r), 2),  # 回合奖励 3
                                                        self.current_time_step,  # makespan 4
                                                        round(cost, 2),  # energy 5
                                                        round(self.current_time_step / self.max_t,2),  # scaled makespan 6
                                                        round(cost / self.max_c, 2),  # scaled energy 7
                                                        round(-scaled_reward, 2),  # objective (final reward) 8
                                                        self.hole_plannings,  # idle machine times 9
                                                        round(epi_time, 2)), file=f3)  # episode running time 10
            self.epi_r.append(scaled_reward)
            return (
                self._get_current_state_representation(),
                scaled_reward,
                self._is_done(),
                {},
            )

    def _reward_scaler(self, reward):
        return reward / self.max_time_op

    def increase_time_step(self):
        """
        The heart of the logic his here, we need to increase every counter when we have a nope action called
        and return the time elapsed
        :return: time elapsed
        """
        hole_planning = 0
        next_time_step_to_pick = self.next_time_step.pop(0)
        self.next_jobs.pop(0)
        difference = next_time_step_to_pick - self.current_time_step
        self.current_time_step = next_time_step_to_pick
        for job in range(self.jobs):
            was_left_time = self.time_until_finish_current_op_jobs[job]
            if was_left_time > 0:
                performed_op_job = min(difference, was_left_time)
                self.time_until_finish_current_op_jobs[job] = max(
                    0, self.time_until_finish_current_op_jobs[job] - difference
                )
                self.state[job][1] = (
                    self.time_until_finish_current_op_jobs[job] / self.max_time_op
                )
                self.total_perform_op_time_jobs[job] += performed_op_job
                self.state[job][3] = (
                    self.total_perform_op_time_jobs[job] / self.max_time_jobs
                )
                if self.time_until_finish_current_op_jobs[job] == 0:  # 这次的时间驱动刚好完成了一个操作
                    self.total_idle_time_jobs[job] += difference - was_left_time
                    self.state[job][6] = self.total_idle_time_jobs[job] / self.sum_op
                    self.idle_time_jobs_last_op[job] = difference - was_left_time
                    self.state[job][5] = self.idle_time_jobs_last_op[job] / self.sum_op
                    self.todo_time_step_job[job] += 1
                    self.state[job][2] = self.todo_time_step_job[job] / self.machines
                    if self.todo_time_step_job[job] < self.machines:
                        self.needed_machine_jobs[job] = self.instance_matrix[job][
                            self.todo_time_step_job[job]
                        ][0]
                        self.state[job][4] = (
                            max(
                                0,
                                self.time_until_available_machine[
                                    self.needed_machine_jobs[job]
                                ]
                                - difference,
                            )
                            / self.max_time_op
                        )
                    else:
                        self.needed_machine_jobs[job] = -1
                        # this allow to have 1 is job is over (not 0 because, 0 strongly indicate that the job is a
                        # good candidate)
                        self.state[job][4] = 1.0
                        if self.legal_actions[job]:
                            self.legal_actions[job] = False
                            self.nb_legal_actions -= 1
            elif self.todo_time_step_job[job] < self.machines:
                self.total_idle_time_jobs[job] += difference
                self.idle_time_jobs_last_op[job] += difference
                self.state[job][5] = self.idle_time_jobs_last_op[job] / self.sum_op
                self.state[job][6] = self.total_idle_time_jobs[job] / self.sum_op
        for machine in range(self.machines):
            if self.time_until_available_machine[machine] < difference:
                empty = difference - self.time_until_available_machine[machine]
                hole_planning += empty
            self.time_until_available_machine[machine] = max(
                0, self.time_until_available_machine[machine] - difference
            )
            if self.time_until_available_machine[machine] == 0:
                for job in range(self.jobs):
                    if (
                        self.needed_machine_jobs[job] == machine
                        and not self.legal_actions[job]
                        and not self.illegal_actions[machine][job]
                    ):
                        self.legal_actions[job] = True
                        self.nb_legal_actions += 1
                        if not self.machine_legal[machine]:
                            self.machine_legal[machine] = True
                            self.nb_machine_legal += 1
        return hole_planning

    def _is_done(self):
        if self.nb_legal_actions == 0:
            self.last_time_step = self.current_time_step
            self.last_solution = self.solution
            return True
        return False

    # 在每个episode结束后计算消耗的能量; 可能会报logic error?
    def calculate_energy(self):
        assert self._is_done() is True, 'The episode is not done!!'
        # 先把self.solution处理一下
        p_priority = self.instance_matrix[:, :, 0]
        p_time = self.instance_matrix[:, :, 1]
        solution = np.array(self.solution)
        solution_end = solution + p_time

        Q_start_ = np.zeros_like(solution)
        Q_end_ = np.zeros_like(solution)
        for job_i in range(solution.shape[0]):
            for o_j, needed_machine_index in enumerate(p_priority[job_i]):
                Q_start_[job_i][needed_machine_index] = solution[job_i][o_j]
                Q_end_[job_i][needed_machine_index] = solution_end[job_i][o_j]

        Q_start = Q_start_.T
        Q_end = Q_end_.T

        Q_start_sorted = np.zeros_like(Q_start)
        Q_end_sorted = np.zeros_like(Q_end)

        arg_sort = np.argsort(Q_start, axis=1)
        for machine_i in range(arg_sort.shape[0]):
            for j, job_i in enumerate(arg_sort[machine_i]):
                Q_start_sorted[machine_i][j] = Q_start[machine_i][job_i]
                Q_end_sorted[machine_i][j] = Q_end[machine_i][job_i]

        C = 0

        # 下面的参数需要指派
        sigma_machine = [self.sigma_machine[j] for j in range(Q_start.shape[0])]
        T_warmup_machine = [self.T_warmup_machine[j] for j in range(Q_start.shape[0])]
        u_k_p = self.u_k_p
        u_k_w = self.u_k_w

        max_diff = 0
        for machine_k in range(Q_start_sorted.shape[0]):
            for q in range(0, Q_start_sorted.shape[1]-1, 1):
                assert Q_start_sorted[machine_k][q+1] - Q_end_sorted[machine_k][q] >= 0, 'logic error'
                max_diff = max(max_diff, Q_start_sorted[machine_k][q+1] - Q_end_sorted[machine_k][q])
                if Q_start_sorted[machine_k][q+1] - Q_end_sorted[machine_k][q] > sigma_machine[machine_k]:  # 停机warmup
                    C += u_k_w * T_warmup_machine[machine_k]
                    # print(1, u_k_w * T_warmup_machine[machine_k], Q_start_sorted[machine_k][q+1] - Q_end_sorted[machine_k][q])
                else:
                    # print(2, u_k_p * (Q_start_sorted[machine_k][q+1] - Q_end_sorted[machine_k][q]))
                    C += u_k_p * (Q_start_sorted[machine_k][q+1] - Q_end_sorted[machine_k][q])
        self.c = C
        return C

    def render(self, mode="human"):
        df = []
        for job in range(self.jobs):
            i = 0
            while i < self.machines and self.solution[job][i] != -1:
                dict_op = dict()
                dict_op["Task"] = "Job {}".format(job)
                start_sec = self.start_timestamp + self.solution[job][i]
                finish_sec = start_sec + self.instance_matrix[job][i][1]
                dict_op["Start"] = datetime.datetime.fromtimestamp(start_sec)
                dict_op["Finish"] = datetime.datetime.fromtimestamp(finish_sec)
                dict_op["Resource"] = "Machine {}".format(
                    self.instance_matrix[job][i][0]
                )
                df.append(dict_op)
                i += 1
        fig = None
        if len(df) > 0:
            df = pd.DataFrame(df)
            fig = ff.create_gantt(
                df,
                index_col="Resource",
                colors=self.colors,
                show_colorbar=True,
                group_tasks=True,
            )
            fig.update_yaxes(
                autorange="reversed"
            )  # otherwise tasks are listed from the bottom up
        return fig

    def dynamic_processing_time(self):

        assert sum(self.todo_time_step_job) < self.machines * len(self.todo_time_step_job), 'already finished !!'

        may_changed_jobs = []
        for job_i in range(len(self.todo_time_step_job)):
            if self.todo_time_step_job[job_i] < (self.machines / 2):
                may_changed_jobs.append(job_i)
        if len(may_changed_jobs) == 0:
            return
        random_select = random.randint(0, len(may_changed_jobs)-1)
        chosen_job = may_changed_jobs[random_select]

        if len(self.dynamic_changed_jobs) >= int(self.jobs/5 + 1):
            return  # 不需要出现太多次的动态变化

        if chosen_job in self.dynamic_changed_jobs:
            return
        else:
            self.dynamic_changed_jobs.append(chosen_job)

        avail_ops = list(range(self.todo_time_step_job[chosen_job]+1, self.machines-1, 1))

        if len(avail_ops) == 0:
            return
        random_select2 = random.randint(0, len(avail_ops)-1)
        chosen_op = avail_ops[random_select2]

        original_time = self.instance_matrix[chosen_job][chosen_op][1]
        chosen_machine = self.instance_matrix[chosen_job][chosen_op][0]

        u_T_wup = self.T_warmup_machine[chosen_machine]

        added_time = int(random.normalvariate(u_T_wup, 1))
        # 下面这一句话修改了全局的变量:
        self.instance_matrix[chosen_job][chosen_op][1] = original_time + added_time

