## Introduction

The light codes for the paper published in IJPR.

This paper proposes to solve the dynamic job scheduling in the context of CCMS with a novel federated deep reinforcement learning (FDRL) approach. To handle heterogeneous policy structures, we aggregate their hidden parameters through FDRL, with states, actions, and rewards designed to facilitate the aggregation. The two-phase algorithm, comprising iterative local training and global aggregation, trains the scheduling policies. Constraint items are introduced to the loss functions to smooth local training, and the global aggregation considers production scales and obtained objectives. The proposed approach enhances the solution quality and generalization of each factory's scheduling policy without exposing original production data. Numerical experiments conducted on sixty scheduling instances validate the superiority of the proposed approach compared to twelve dynamic scheduling methods. Compared to independently trained DRL-based approaches, the proposed FDRL-based approach achieves up to an 8.9\% reduction in makespan and a 22.3\% decrease in energy consumption through knowledge sharing.

If you find something valuable (maybe), please cite our work as:

@article{wang2024federated,
  title={Federated Deep Reinforcement Learning for Dynamic Job Scheduling in Cloud-edge Collaborative Manufacturing Systems},
  author={Wang, Xiaohan and Zhang, Lin and Wang, Lihui and Wang, Xi Vincent and Liu, Yongkui},
  journal={International Journal of Production Research},
  year={2024},
  publisher={Taylor \& Francis}
}

## Implementation
I have simplified the original codes and reserve the core components for execution to make the codes clear and easy to reuse. The local scheduling policies are trained based on multi-CPU to accelerate the training. To start, just run:
```python
python main.FPPO.py
```
