
import gymnasium as gym
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import random
import optax

N_OBSERVATIONS = 8
N_ACTIONS = 4 # change if im using other environments in future
EPSILON = 0.3
BATCH_SIZE = 3
GAMMA = 0.9
LEARNING_RATE = 0.1

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

class NeuralNetwork(eqx.Module):
    layers: list
    extra_bias: jax.Array
    def __init__(self, key):
        key1, key2, key3 = jax.random.split(key, 3)
        # These contain trainable parameters.
        self.layers = [ eqx.nn.Linear(N_OBSERVATIONS, 64, key=key1),
                        eqx.nn.Linear(64, 64, key=key2),
                        eqx.nn.Linear(64, N_ACTIONS, key=key3) ]
        # This is also a trainable parameter.
        self.extra_bias = jax.numpy.ones(4)

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x) + self.extra_bias

### ------------- INITIAL INITIALIZATIONS --------------------------------------------------------

# Initialize model
eps_key, env_key, model_key = jax.random.split(jax.random.PRNGKey(0), 3)

# Initialize environment
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)

buffer = []

### ------------- FUNCTIONS --------------------------------------------------------

@jax.jit # compile this function to make it run fast.
@jax.grad # differentiate all floating-point arrays in `model`.
def loss(expected, target):
    return jax.numpy.mean((target - expected) ** 2) # L2 loss

def select_action(obs, epsilon, model):
    """
    Choose action, a random one or purposeful one
    """
    if np.random.uniform()< epsilon:
        return np.random.randint(0,3)
    else:
        return int(jnp.argmax(model(obs)))


def optimize_model(model, target_model):

    # only start optimizing once our buffer size is big enough
    if BATCH_SIZE+10 > len(buffer):
        return model, target_model
    
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
    # Calculate target value from bellman equation
    q_targets = batch_rewards + GAMMA * q_targets_next * (1 - batch_dones)
    # Calculate expected value from local network
    q_expected = jax.vmap(model)(jnp.array(batch_states))
    print(q_expected)
    q_expected_a = jnp.array([q_expected[i][batch_actions[i]] for i in range(len(q_expected))])

    """Grad output isn't the size/shape needed to go into the optimizer"""
    ### Loss calculation (we used L2 Loss)
    grads = loss(q_expected_a, q_targets)
    # print(grads)
    # Define the optimizer (e.g., adam)

    """TODOOOO: just add the optimizer???"""

    """Also update target network? soft update"""

    return 


### ------------------------------------------------------------------------------------------

model = NeuralNetwork(model_key)
target_model = NeuralNetwork(model_key)

# LOOP
for _ in range(BATCH_SIZE+10):
    action = select_action(observation, EPSILON, model) 
    observation_prev = observation
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
    
    buffer.append([observation_prev, action, reward, observation, terminated])

    model, target_model = optimize_model(model, target_model)
        
    # grads = loss(model, observation, reward)
print("COMPLETE!")
env.close()
