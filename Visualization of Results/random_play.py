import numpy as np, os,random
from core import mod_utils as utils
from core.ddpg import GeneticAgent
from parameters import Parameters
import torch
import gym
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] ='1'
parser = argparse.ArgumentParser()
parser.add_argument('-env', help='Environment Choices: (HalfCheetah-v2) (Ant-v2) (Reacher-v2) (Walker2d-v2) ' +
                                 '(Swimmer-v2) (Hopper-v2)', required=True, type=str)
parser.add_argument('-render', help='Render gym episodes', action='store_true')
parser.add_argument('-model_path', help='Path to the model', type=str, required=True)
args = parser.parse_args()


def evaluate(agent, env, trials=30, render=False, seed=None):
    results = []
    for trial in range(trials):
        total_reward = 0

        state = env.reset()
        done = False
        while not done:
            if render: env.render()
            action = agent.actor.select_action(np.array(state))

            # Simulate one step in environment
            next_state, reward, done, info = env.step(action.flatten())
            total_reward += reward
            state = next_state

        results.append(total_reward) 

    print('Test seed: ', seed, "Average score over 30 episodes: ", np.mean(results))


def load_genetic_agent(args):
    actor_path = os.path.join(args.model_path)
    agent = GeneticAgent(args)
    agent.actor.load_state_dict(torch.load(actor_path))

    return agent


if __name__ == "__main__":
    env = utils.NormalizedActions(gym.make(args.env))
    parameters = Parameters(None, init=False)
    parameters.individual_bs = 0
    parameters.action_dim = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]
    parameters.use_ln = True
    parameters.device = torch.device('cuda')
    parameters.ls=64
    setattr(parameters, 'model_path', args.model_path)
    agent = load_genetic_agent(parameters)  
    for _ in range(50):
        seed = np.random.randint(10000)
        env.seed(seed)
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        evaluate(agent, env, render=args.render, seed=seed)
