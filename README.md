# Surrogate-assisted Evolutionary Reinforcement Learning

**Abstract**

The integration of Reinforcement Learning (RL) and Evolutionary Algorithms (EAs) aims at simultaneously exploiting the sample efficiency as well as the diversity and robustness of the two paradigms. Recently, hybrid learning frameworks based on this principle have achieved great success in robot control tasks. However, in these methods, policies from the genetic population are evaluated via interactions with the real environments, severely restricting their applicability when such interactions are prohibitively costly. In this work, we propose Surrogate-assisted Controller (SC), a generic module that can be applied on top of existing hybrid frameworks to alleviate the computational burden of expensive fitness evaluation. The key to our approach is to leverage the critic network that is implemented in existing hybrid frameworks as a novel surrogate model, making it possible to estimate the fitness of individuals without environmental interactions. In addition, two model management strategies with the elite protection mechanism are introduced in SC to control the workflow, leading to a fast and stable optimization process. In the empirical studies, we combine SC with two state-of-the-art evolutionary reinforcement learning approaches to highlight its functionality and effectiveness. Experiments on six challenging continuous control benchmarks from the OpenAI Gym platform show that SC can not only significantly reduce the cost of interaction with the environment, but also bring better sample efficiency and dramatically boost the learning progress of the original hybrid framework.

![image](serl.jpg)


