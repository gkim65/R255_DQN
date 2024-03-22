
################### LIBRARIES ######################

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import pygame
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

################### GPU AND MATPLOTLIB ######################

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print (x)
else:
    print ("MPS device not found.")
    device = torch.device("cpu")

################### INITIAL PARAMETERS ######################

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.5
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR =  1e-4
REPLAYMEMORY = 10000


test_name = "Test_6"
# Things about env:
max_episode_steps = 500
# Things to change for domain randomization:
render_mode = "human" #None
x_threshold = [.5,4.8]
length = [.1,1]# actually half the pole's length 0.5
masscart = [0.1,1]
masspole = [0.1,1]
force_mag = 10
tau = 0.02 # seconds between state updates
theta_threshold_radians = 24 * 2 * math.pi / 360

# Data for later:
episode_durations = []
losses = []
observation_log = []
reward_log = []

if torch.backends.mps.is_available():
    num_episodes = 500
else:
    num_episodes = 15

#################### ENVIRONMENT ###############################

env = gym.make("CartPole-v1", render_mode = render_mode,
                x_threshold = np.random.uniform(x_threshold[0], x_threshold[1]),
                length = np.random.uniform(length[0], length[1]),
                masscart = np.random.uniform(masscart[0], masscart[1]),
                masspole = np.random.uniform(masspole[0], masspole[1]),
                force_mag = force_mag,
                tau = tau,
                theta_threshold_radians = theta_threshold_radians
                )

print(env.masscart)
# Get number of actions from gym action space
n_actions = env.action_space.n

# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

#################### REPLAY MEMORY ###############################

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


###################### NEURAL NETWORK ##############################

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

####################### SETUP OF OPTIMIZERS AND NN #############################

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(REPLAYMEMORY)

####################### FUNCTIONS #############################

# Select action, randomness of action selected by epsilon, or max of values 
def select_action(state, steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


# Plot the training progress as we move forward
def plot_durations(episode_durations,show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# Select action, randomness of action selected by epsilon, or max of values 
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. 
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    losses.append(loss.item())

    # Optimize the model
    optimizer.zero_grad()
    
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

####################### MAIN TRAINING LOOP #############################

steps_done = 0
for i_episode in range(num_episodes):
    reward_total = 0
    # Initialize the environment and get it's state
    env = gym.make("CartPole-v1", render_mode = render_mode,
                x_threshold = np.random.uniform(x_threshold[0], x_threshold[1]),
                length = np.random.uniform(length[0], length[1]),
                masscart = np.random.uniform(masscart[0], masscart[1]),
                masspole = np.random.uniform(masspole[0], masspole[1]),
                force_mag = force_mag,
                tau = tau,
                theta_threshold_radians = theta_threshold_radians
                )
    state, info = env.reset()

    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state, steps_done)
        steps_done += 1
        observation, reward, terminated, truncated, _ = env.step(action.item())

        if i_episode == num_episodes-1:
            observation_log.append([observation[0],observation[1],observation[2],observation[3],action.item()])
        
        if 3 > 0:
            for i in range(1):
                observation, reward, terminated, truncated, _ = env.step(action.item())

                reward = torch.tensor([reward], device=device)
                done = terminated or truncated
                if terminated:
                    next_state = None
                    break
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                    memory.push(state, action, next_state, reward)
                    # Move to the next state
                    state = next_state
        else:
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            if terminated:
                next_state = None
                break
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        reward_total = reward_total + reward
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            reward_log.append(reward_total.item())
            episode_durations.append(t + 1)
            plot_durations(episode_durations)
            break

print('Complete')
plot_durations(episode_durations, show_result=True)
plt.ioff()
plt.show()
pygame.quit()
env.close()


####################### SAVE DATA #############################


data_folder_name = 'results/'+test_name+"_bs"+str(BATCH_SIZE) +"_g" +str(GAMMA)+"_epSt" +str(EPS_START) +"_epEn" + str(EPS_END) + "_epDe"+ str(EPS_DECAY) + "_t" + str(TAU) + "_LR" + str(LR) + "_rm" +str(REPLAYMEMORY)
if not os.path.exists(data_folder_name):
    os.makedirs(data_folder_name)

x = pd.DataFrame(episode_durations, columns=["episode_durations"])
x.to_csv(data_folder_name+"/episode_durations.csv")

x = pd.DataFrame(losses, columns=["losses"])
x.to_csv(data_folder_name+"/losses.csv")

x = pd.DataFrame(observation_log, columns=["Position_Cart", "Velocity_Cart", "Pole_Angle", "Pole_Angular_Velocity", "Action_Taken"])
x.to_csv(data_folder_name+"/observation_log.csv")

x = pd.DataFrame(reward_log, columns=["Rewards"])
x.to_csv(data_folder_name+"/rewards.csv")

params = {"Batch_size": BATCH_SIZE, 
        "Gamma": GAMMA,
        "EPS_START": EPS_START,
        "EPS_END": EPS_END,
        "EPS_DECAY": EPS_DECAY,
        "TAU": TAU,
        "Learning Rate": LR,
        "n_actions": n_actions,
        "optimizer": "adam",
        "num_episodes": num_episodes,
        "test_name" : test_name,
        "max_episode_steps" : max_episode_steps,
        "render_mode" : render_mode,
        "x_threshold" : x_threshold,
        "length" : length,
        "masscart" : masscart,
        "masspole" : masspole,
        "force_mag" : force_mag,
        "tau" : tau,
        "theta_threshold_radians" : theta_threshold_radians
        }

x = pd.DataFrame(params) #, index=[0])
x.to_csv(data_folder_name+"/params.csv")

torch.save(policy_net.state_dict(),data_folder_name+"/cart_pole_policy.pt")
torch.save(target_net.state_dict(),data_folder_name+"/cart_pole_target.pt")
