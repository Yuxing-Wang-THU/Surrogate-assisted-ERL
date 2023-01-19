import numpy as np, os, time, random
from core import mod_neuro_evo as utils_ne
from core import mod_utils as utils
import gym, torch
from core import replay_memory
from core import ddpg as ddpg
import argparse
from copy import deepcopy

# Parsers
render = False
parser = argparse.ArgumentParser()
parser.add_argument('--env', help='Environment Choices: (HalfCheetah-v2) (Ant-v2) (Reacher-v2) (Walker2d-v2) (Swimmer-v2) (Hopper-v2)', default='Hopper-v2')
parser.add_argument('--seed', help='seed', default=1)
parser.add_argument('--surate', help='Probability of using surrogate-assisted evaluation', type=float, default=0.6)
parser.add_argument('--deviceid',help='GPU device ID', default=1)

env_tag = vars(parser.parse_args())['env']
seed = int(vars(parser.parse_args())['seed'])
surro_fit_rate = float(vars(parser.parse_args())['surate'])
real_fit_rate = 1 - surro_fit_rate

# Set GPU 
device = str(vars(parser.parse_args())['deviceid'])
os.environ["CUDA_VISIBLE_DEVICES"] = device

# Print
print("---------------------------------------------------")
print("Env name: ", env_tag)
print("Seed: ", seed)
print("Surro rate: ", surro_fit_rate)
print("Real rate: ", real_fit_rate)
print("GPU device: ", os.environ["CUDA_VISIBLE_DEVICES"] )
print("---------------------------------------------------")

# ERL Parameters
class Parameters:
    def __init__(self):
        # Number of Frames to Run
        if env_tag == 'Ant-v2' or env_tag == 'HalfCheetah-v2':
            self.num_frames = 6000000
        else: 
            self.num_frames = 3000000

        # Use CUDA
        self.is_cuda = True; self.is_memory_cuda = True
        
        # Set the device to run on CUDA or CPU
        if self.is_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Sync Period
        if env_tag == "Swimmer-v2":
            self.synch_period = 10
        else: 
            self.synch_period = 1

        # DDPG params
        self.use_ln = True
        self.gamma = 0.99; self.tau = 0.001
        self.seed = seed
        self.batch_size = 128
        self.buffer_size = 1000000
        self.frac_frames_train = 1.0
        self.use_done_mask = True

        ###### NeuroEvolution Params ########
        # Num of trials
        if env_tag == 'Hopper-v2' or env_tag == 'Reacher-v2': self.num_evals = 5
        elif env_tag == 'Walker2d-v2': self.num_evals = 3
        else: self.num_evals = 1

        # Elitism Rate
        if env_tag == 'Hopper-v2' or env_tag == 'Ant-v2': self.elite_fraction = 0.3
        elif env_tag == 'Reacher-v2' or env_tag == 'Walker2d-v2': self.elite_fraction = 0.2
        else: self.elite_fraction = 0.1

        self.pop_size = 10
        self.crossover_prob = 0.0
        self.mutation_prob = 0.9

        # Generation-based control
        self.surro_fit_rate = surro_fit_rate
        self.real_fit_rate = real_fit_rate
        self.evaluation_memory_size = 50000
        self.surro_batch_size = 1024

        # Logs folder
        self.state_dim = None; self.action_dim = None #Simply instantiate them here, will be initialized later
        self.save_foldername = env_tag +'/'+ f'SERL-G-{surro_fit_rate}-{seed}/' 
        if not os.path.exists(self.save_foldername): 
            os.makedirs(self.save_foldername)

# SERL Agent
class Agent:
    def __init__(self, args, env):
        self.args = args; self.env = env
        self.evolver = utils_ne.SSNE(self.args)
        self.best_actor = None
        
        # Parameters of SC
        self.eval_memory_size = self.args.evaluation_memory_size
        self.surro_fit_rate = self.args.surro_fit_rate
        self.real_fit_rate = self.args.real_fit_rate
        self.surro_critic = None

        # Init population
        self.pop = []
        for _ in range(args.pop_size):
            self.pop.append(ddpg.Actor(args))

        # Turn off gradients and put in eval mode
        for actor in self.pop: actor.eval()

        # Init RL Agent
        self.rl_agent = ddpg.DDPG(args)
        self.replay_buffer = replay_memory.ReplayMemory(args.buffer_size)
        self.ounoise = ddpg.OUNoise(args.action_dim)

        # Trackers
        self.num_games = 0; self.num_frames = 0; self.gen_frames = None
   
    # Functions of Surrogate-assisted Controller
    # --------------------------------------------------------
    def select(self):
        num_ = ['surro', 'real']
        r_ = [self.surro_fit_rate, self.real_fit_rate]
        sum_ = 0
        ran = random.random()
        for num, r in zip(num_, r_):
            sum_ += r
            if ran < sum_ :break
        return num
    
    def exchange_para(self, source_net, target_net):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(param.data)
    
    def surro_evaluate(self, net, memory):
        fitness = 0
        datas = replay_memory.Transition(*zip(*memory))
        states = torch.cat(datas.state)
        n = len(states)
        arr = np.arange(n)
        for i in range(n // self.args.surro_batch_size):
            batch_index = arr[self.args.surro_batch_size * i: self.args.surro_batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index) 
            inputs = states[batch_index]
            actions = net(inputs)
            surro_rewards = self.surro_critic(inputs, actions)
            fitness += np.sum(surro_rewards.data.cpu().numpy())
        return fitness / n
    
    def add_experience(self, state, action, next_state, reward, done):
        reward = utils.to_tensor(np.array([reward])).unsqueeze(0)
        if self.args.is_cuda: reward = reward.cuda()
        if self.args.use_done_mask:
            done = utils.to_tensor(np.array([done]).astype('uint8')).unsqueeze(0)
            if self.args.is_cuda: done = done.cuda()
        action = utils.to_tensor(action)
        if self.args.is_cuda: action = action.cuda()
        self.replay_buffer.push(state, action, next_state, reward, done)

    def evaluate(self, net, is_render=False, is_action_noise=False, store_transition=True):
        total_reward = 0.0
        state = self.env.reset()
        state = utils.to_tensor(state).unsqueeze(0)
        if self.args.is_cuda: state = state.cuda()
        done = False

        while not done:
            if store_transition: self.num_frames += 1; self.gen_frames += 1
            if render and is_render: self.env.render()
            action = net.forward(state)
            action.clamp(-1,1)
            action = utils.to_numpy(action.cpu())
            if is_action_noise: action += self.ounoise.noise()
            next_state, reward, done, info = self.env.step(action.flatten())  #Simulate one step in environment
            next_state = utils.to_tensor(next_state).unsqueeze(0)
            if self.args.is_cuda:
                next_state = next_state.cuda()
            total_reward += reward
            if store_transition: 
                self.add_experience(state, action, next_state, reward, done)
            state = next_state
        if store_transition: self.num_games += 1
        return total_reward

    def rl_to_evo(self, rl_net, evo_net):
        for target_param, param in zip(evo_net.parameters(), rl_net.parameters()):
            target_param.data.copy_(param.data)
                        
    def train(self):
        self.gen_frames = 0
        ####################### EVOLUTION #####################
        all_fitness = []
        # Evaluate genomes/individuals
        butt = self.select()
        if len(self.replay_buffer) <=1000:
            butt = "real"
        if butt == "real":
            print("######## Real Fitness Evalution ###########")
            for net in self.pop:
                fitness = 0.0
                for eval in range(self.args.num_evals): fitness += self.evaluate(net, is_render=False, is_action_noise=False)
                all_fitness.append(fitness/self.args.num_evals)
            
            # Validation test
            champ_index = all_fitness.index(max(all_fitness))
            best_train_fitness = max(all_fitness)
            worst_index = all_fitness.index(min(all_fitness))
            test_score = 0.0
            
            # For Elite protection
            self.best_actor = deepcopy(self.pop[champ_index])
            
            # Report best fitness
            for eval in range(5): test_score += self.evaluate(self.best_actor, is_render=False, is_action_noise=False, store_transition=False)/5.0
            
            # NeuroEvolution's probabilistic selection and recombination step
            elite_index, all_elites, unselects = self.evolver.epoch(self.pop, all_fitness, butt=butt)
        
        elif butt == 'surro':
            print("######## Surrogate-assisted Evaluation ###########")
            # Prepare critic-based surrogate model
            self.surro_critic = deepcopy(self.rl_agent.critic)
            # Prepare evaluation memory
            if len(self.replay_buffer) < self.eval_memory_size:
                evaluation_memory = self.replay_buffer.memory
            else:
                evaluation_memory = self.replay_buffer.memory[-self.eval_memory_size::]
            for net in self.pop:
                fitness = self.surro_evaluate(net, memory=evaluation_memory)
                all_fitness.append(fitness)

            test_score = None
            best_train_fitness = max(all_fitness)
            worst_index = all_fitness.index(min(all_fitness))
            champ_index = all_fitness.index(max(all_fitness))
            # NeuroEvolution's probabilistic selection and recombination step
            elite_index,_,_ = self.evolver.epoch(self.pop, all_fitness, butt=butt,)
            # Elite protection
            self.exchange_para(self.best_actor, self.pop[worst_index])

        ####################### DDPG Part#########################
        #DDPG Experience Collection
        rl_score = self.evaluate(self.rl_agent.actor, is_render=False, is_action_noise=True) #Train
    
        # Validation test for RL agent
        testr = 0
        for eval in range(5):
            testr += self.evaluate(self.rl_agent.actor, is_render=False, store_transition=False, is_action_noise=False)/5
        
        # DDPG learning step
        if len(self.replay_buffer) > self.args.batch_size * 5:
            for _ in range(int(self.gen_frames*self.args.frac_frames_train)):
                transitions = self.replay_buffer.sample(self.args.batch_size)
                batch = replay_memory.Transition(*zip(*transitions))
                self.rl_agent.update_parameters(batch)

            # Synch RL Agent to NE and butt == "real"
            # Elite protection
            if self.num_games % self.args.synch_period == 0 and butt == "real":
                if worst_index not in all_elites:
                    self.rl_to_evo(self.rl_agent.actor, self.pop[worst_index])
                    self.evolver.rl_policy = worst_index
                    self.evolver.best_real_policy = champ_index
                    print('Synch from RL --> Nevo')
                else:
                    if len(unselects) > 0:
                        self.rl_to_evo(self.rl_agent.actor, self.pop[unselects[-1]])
                        self.evolver.rl_policy = unselects[-1]
                        self.evolver.best_real_policy = champ_index
                        print('Synch from RL --> Nevo')
                    else:
                        self.rl_to_evo(self.rl_agent.actor, self.pop[all_elites[-1]])
                        self.evolver.rl_policy = all_elites[-1]
                        self.evolver.best_real_policy = champ_index
                        print('Synch from RL --> Nevo')
                
        if test_score is not None:
            return best_train_fitness, test_score, elite_index
        else:
            return 0, 0, 0

if __name__ == "__main__":
    parameters = Parameters()
    tracker = utils.Tracker(parameters, ['serl'], '_score.csv')  
    frame_tracker = utils.Tracker(parameters, ['frame_serl'], '_score.csv')  
    time_tracker = utils.Tracker(parameters, ['time_serl'], '_score.csv')

    # Create Env
    env = utils.NormalizedActions(gym.make(env_tag))
    parameters.action_dim = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]

    # Seed
    env.seed(parameters.seed)
    torch.manual_seed(parameters.seed); np.random.seed(parameters.seed); random.seed(parameters.seed)

    # Create SERL Agent
    agent = Agent(parameters, env)

    next_save = 100; time_start = time.time()
    
    generation = 0
    
    # Begin training
    while agent.num_frames <= parameters.num_frames:
        best_train_fitness, erl_score, elite_index = agent.train()
        generation +=1
        if best_train_fitness != 0 and erl_score != 0:
            print('#Games:', agent.num_games, '#Frames:', agent.num_frames, ' Epoch_Max:', '%.2f'%best_train_fitness if best_train_fitness != None else None, ' Test_Score:','%.2f'%erl_score if erl_score != None else None, ' Avg:','%.2f'%tracker.all_tracker[0][1], 'ENV '+env_tag)
            tracker.update([erl_score], agent.num_games)
            frame_tracker.update([erl_score], agent.num_frames)
            time_tracker.update([erl_score], time.time()-time_start)
        #Save Policy
        if agent.num_games > next_save:
            next_save += 100
            if elite_index != None: torch.save(agent.pop[elite_index].state_dict(), parameters.save_foldername + 'evo_net.pkl')
            print("Progress Saved !!!!!")






