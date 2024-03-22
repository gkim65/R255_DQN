
import gymnasium as gym
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import random
import optax
import matplotlib.pyplot as plt
import math
from optax.tree_utils import tree_scalar_mul,tree_add

N_OBSERVATIONS = 8
N_ACTIONS = 4 # change if im using other environments in future
BATCH_SIZE = 128
GAMMA = 0.99
LEARNING_RATE =  2.5e-4
TOTAL_STEPS = 1001
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
START_TRAIN = 1000


"""

GENERAL OUTLINE/FLOW:

1) Initialize the model and environments
2) Select an action
    - Action selected from random with probability, or action selected using
    - Argmax from the actions given by nn
3) Execute action, get observations and see if we stop
4) Add action into REPLAY BUFFER
5) Compute losses using target and policy networks, use states from reply buffer
    5a) compute gradients, then backpropagate for optimizer using optax

"""



### ------------- NEURAL NETWORK --------------------------------------------------------

class NeuralNetwork(eqx.Module):
    layers: list
    extra_bias: jax.Array
    def __init__(self, key):
        key1, key2, key3 = jax.random.split(key, 3)
        # These contain trainable parameters.
        self.layers = [ eqx.nn.Linear(N_OBSERVATIONS, 120, key=key1),
                        eqx.nn.Linear(120, 64, key=key2),
                        eqx.nn.Linear(64, N_ACTIONS, key=key3) ]
        # This is also a trainable parameter.
        self.extra_bias = jax.numpy.ones(4)

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x) + self.extra_bias
    
    def multiply_by_scalar(self, scalar):
        for layer in self.layers:
            for param in layer.weight:
                param *= scalar

### ------------- INITIAL INITIALIZATIONS --------------------------------------------------------

# Initialize model
eps_key, env_key, model_key = jax.random.split(jax.random.PRNGKey(0), 3)

# Initialize environment
env = gym.make("LunarLander-v2") #, render_mode="human")
observation, info = env.reset(seed=42)

buffer = []

### ------------- FUNCTIONS --------------------------------------------------------

# @jax.jit # compile this function to make it run fast.
# @jax.grad # differentiate all floating-point arrays in `model`

def select_action(obs, model, steps_done):
    """
    Choose action, a random one or purposeful one
    """
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)

    if START_TRAIN > len(buffer) and np.random.uniform() < eps_threshold:
        return np.random.randint(0,4)
    else:
        return int(jnp.argmax(model(obs)))


def optimize_model(model, target_model, optimizer, opt_state):
    
    # only start optimizing once our buffer size is big enough
    if START_TRAIN > len(buffer):
        return model, target_model, 0, opt_state,0
    
    # make a batch to train with
    rand_ind = random.sample(list(range(len(buffer))),k = BATCH_SIZE)
    batch = [buffer[ind] for ind in rand_ind]

    ### BELLMAN EQUATION
    batch_next_states = jnp.array([row[3] for row in batch])
    batch_rewards = jnp.array([row[2] for row in batch])
    batch_dones = jnp.array([row[4] for row in batch])
    batch_actions = jnp.array([row[1] for row in batch])
    batch_states = jnp.array([row[0] for row in batch])

    # find largest Q value possible within next steps
    q_targets_next = jnp.max(jax.vmap(target_model)(jnp.array(batch_next_states)),1)
    q_targets = batch_rewards + GAMMA * q_targets_next * (1 - batch_dones)

    def loss(model, x, target, actions):
        # Calculate expected value from local network
        q_expected = jax.vmap(model)(jnp.array(x)) 
        q_expected_a = jnp.array([q[i] for i, q in zip(actions, q_expected)])
        return jax.numpy.mean(((q_expected_a - target) ** 2)) # mse loss

    grad_func = jax.value_and_grad(loss)    
    losses, grads = grad_func(model, batch_states, q_targets, batch_actions)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = optax.apply_updates(model, updates)
    
    target_model = tree_add(tree_scalar_mul(1-TAU,target_model),tree_scalar_mul(TAU,model))
    
    return model, target_model, losses, opt_state, np.mean(q_targets)


### ------------------------------------------------------------------------------------------

model = NeuralNetwork(model_key)
target_model = NeuralNetwork(model_key)

optimizer = optax.adam(LEARNING_RATE)
opt_state = optimizer.init(target_model)
loss_list = []
rewards_list = []
q_targets_list = []

# LOOP

reward_total = 0
for step in range(TOTAL_STEPS):
    action = select_action(observation, model, step) 
    observation_next, reward, terminated, truncated, info = env.step(action)

    reward_total += reward
    if terminated or truncated:
        observation, info = env.reset()
        rewards_list.append(reward_total)
        print("rewards: "+ str(reward_total))
        reward_total = 0
    
    buffer.append([observation, action, reward, observation_next, terminated])

    model, target_model, losses, opt_state, q_targets = optimize_model(model, target_model, optimizer, opt_state)

    loss_list.append(losses)
    q_targets_list.append(q_targets)
    if step%1000 == 0:
        print(step)
    observation = observation_next
    


    

plt.plot(range(len(loss_list[100:])),loss_list[100:])
plt.title("Losses Over Time")
plt.xlabel("Rounds")
plt.ylabel("Losses")
plt.show()
plt.plot(range(len(rewards_list)),rewards_list)
plt.title("Total Rewards for each Simulation")
plt.xlabel("Simulation")
plt.ylabel("Rewards")
plt.show()
plt.plot(range(len(q_targets_list[100:])),q_targets_list[100:])

plt.title("Q_Value Over Time")
plt.xlabel("Rounds")
plt.ylabel("Q_Value")
plt.show()

print("COMPLETE!")
env.close()
