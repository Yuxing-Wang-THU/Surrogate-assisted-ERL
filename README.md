# Surrogate-assisted Evolutionary Reinforcement Learning
Code for the Information Sciences paper [A surrogate-assisted controller for expensive evolutionary reinforcement learning](https://www.sciencedirect.com/science/article/abs/pii/S0020025522012658).

<img src="https://github.com/Yuxing-Wang-THU/Surrogate-assisted-ERL/blob/main/serl.png" div align=right width = "37%" />

## Abstract

The integration of Reinforcement Learning (RL) and Evolutionary Algorithms (EAs) aims at simultaneously exploiting the sample efficiency as well as the diversity and robustness of the two paradigms. Recently, hybrid learning frameworks based on this principle have achieved great success in robot control tasks. However, in these methods, policies from the genetic population are evaluated via interactions with the real environments, severely restricting their applicability when such interactions are prohibitively costly. In this work, we propose Surrogate-assisted Controller (SC), a generic module that can be applied on top of existing hybrid frameworks to alleviate the computational burden of expensive fitness evaluation. The key to our approach is to leverage the critic network that is implemented in existing hybrid frameworks as a novel surrogate model, making it possible to estimate the fitness of individuals without environmental interactions. In addition, two model management strategies with the elite protection mechanism are introduced in SC to control the workflow, leading to a fast and stable optimization process. In the empirical studies, we combine SC with two state-of-the-art evolutionary reinforcement learning approaches to highlight its functionality and effectiveness. Experiments on six challenging continuous control benchmarks from the OpenAI Gym platform show that SC can not only significantly reduce the cost of interaction with the environment, but also bring better sample efficiency and dramatically boost the learning progress of the original hybrid framework.

## Citation

```
@article{wang2022surrogate,
  title={A surrogate-assisted controller for expensive evolutionary reinforcement learning},
  author={Wang, Yuxing and Zhang, Tiantian and Chang, Yongzhe and Wang, Xueqian and Liang, Bin and Yuan, Bo},
  journal={Information Sciences},
  volume={616},
  pages={539--557},
  year={2022},
  publisher={Elsevier}
}
```
