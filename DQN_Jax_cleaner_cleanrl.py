
import gymnasium as gym
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import random
import optax
import matplotlib.pyplot as plt
import math

import flax
import flax.linen as nn
from flax.training.train_state import TrainState
from optax.tree_utils import tree_scalar_mul,tree_add
from stable_baselines3.common.buffers import ReplayBuffer


N_OBSERVATIONS = 8
N_ACTIONS = 4 # change if im using other environments in future
BATCH_SIZE = 128
GAMMA = 0.99
LEARNING_RATE =  2.5e-4
TOTAL_STEPS = 200000
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 1
START_TRAIN = 1000
MAX_BUFFER_SIZE = 10000
TARGET_NET_FREQ = 500

"""

GENERAL OUTLINE/FLOW:

1) Initialize the model and environments
2) Select an action
    - Action selected from random with probability, or action selected using
    - Argmax from the actions given by nn
3) Execute action, get observations and see if we stop
4) Add action into REPLAY BUFFER

Need help starting here:

5) Compute losses using target and policy networks, use states from reply buffer
6) Then optimizer? use optax

"""



### ------------- NEURAL NETWORK --------------------------------------------------------

class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(84)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


class TrainState(TrainState):
    target_params: flax.core.FrozenDict

### ------------- INITIAL INITIALIZATIONS --------------------------------------------------------

# Initialize model
eps_key, env_key, model_key = jax.random.split(jax.random.PRNGKey(0), 3)

def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env.action_space.seed(seed)

        return env

    return thunk

# Initialize environment
env = gym.vector.SyncVectorEnv([make_env("LunarLander-v2", 1)])
# env = gym.make("LunarLander-v2") #, render_mode="human")
observation, info = env.reset(seed=42)

rb = ReplayBuffer(
    MAX_BUFFER_SIZE,
    env.single_observation_space,
    env.single_action_space,
    "cpu",
    handle_timeout_termination=False,
)
### ------------- FUNCTIONS --------------------------------------------------------

def select_action(observation, model, q_state, steps_done):
    """
    Choose action, a random one or purposeful one
    """
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)

    if START_TRAIN > steps_done or np.random.uniform() < eps_threshold:
        return np.array([env.single_action_space.sample() for _ in range(env.num_envs)])

    else:
        q_values = model.apply(q_state.params, observation)
        actions = q_values.argmax(axis=-1)
        return jax.device_get(actions)


def optimize_model(q_state, data, steps_done):
    
    # only start optimizing once our buffer size is big enough
    if START_TRAIN > steps_done:
        return 0,0,q_state,0

    ### BELLMAN EQUATION

    # find largest Q value possible within next steps
    q_next_target = q_network.apply(q_state.target_params, data.next_observations.numpy())  # (batch_size, num_actions)
    q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
    
    # Calculate target value from bellman equation
    next_q_value = data.rewards.flatten().numpy() + (1 - data.dones.flatten().numpy()) * GAMMA * q_next_target

    def mse_loss(params):
        q_pred = q_network.apply(params, data.observations.numpy())  # (batch_size, num_actions)
        q_pred = q_pred[jnp.arange(q_pred.shape[0]), data.actions.numpy().squeeze()]  # (batch_size,)
        return ((q_pred - next_q_value) ** 2).mean(), q_pred

    (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(q_state.params)
    q_state = q_state.apply_gradients(grads=grads)
    return loss_value, q_pred, q_state, next_q_value


### ------------------------------------------------------------------------------------------

q_network = QNetwork(action_dim=env.single_action_space.n)
q_state = TrainState.create(
    apply_fn=q_network.apply,
    params=q_network.init(model_key, observation),
    target_params=q_network.init(model_key, observation),
    tx=optax.adam(learning_rate=LEARNING_RATE),
)

q_network.apply = jax.jit(q_network.apply)
q_state = q_state.replace(target_params=optax.incremental_update(q_state.params, q_state.target_params, 1))

loss_list = []
rewards_list = []
q_targets_list = []

# LOOP

reward_total = 0
for step in range(TOTAL_STEPS):

    action = select_action(observation, q_network, q_state, step)
    observation_next, reward, terminated, truncated, info = env.step(action)

    reward_total += reward
    if terminated or truncated:
        observation, info = env.reset()
        rewards_list.append(reward_total)
        print("rewards: "+ str(reward_total))
        reward_total = 0
    
    rb.add(observation, observation_next, action, reward, terminated, info)

    data = rb.sample(BATCH_SIZE)
    loss_value, q_pred, q_state, next_q_value = optimize_model(q_state, data, step)

    loss_list.append(loss_value)
    q_targets_list.append(np.mean(next_q_value))
    if step%1000 == 0:
        print(step)
        print(np.mean(rewards_list[-20:]))
    observation = observation_next

    if step % TARGET_NET_FREQ == 0:
        q_state = q_state.replace(
            target_params=optax.incremental_update(q_state.params, q_state.target_params, TAU)
        )
    
    if np.mean(rewards_list[-20:]) > 100:
        break

plt.plot(range(len(loss_list[START_TRAIN:])),loss_list[START_TRAIN:])
plt.title("Losses Over Time")
plt.xlabel("Rounds")
plt.ylabel("Losses")
plt.show()
plt.plot(range(len(rewards_list)),rewards_list)
plt.title("Total Rewards for each Simulation")
plt.xlabel("Simulation")
plt.ylabel("Rewards")
plt.show()
plt.plot(range(len(q_targets_list[START_TRAIN:])),q_targets_list[START_TRAIN:])
plt.title("Q_Value Over Time")
plt.xlabel("Rounds")
plt.ylabel("Q_Value")
plt.show()

print("COMPLETE!")
env.close()
