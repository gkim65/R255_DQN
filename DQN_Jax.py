
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

    ### Loss calculation (we used L2 Loss)
    grads = loss(q_expected_a, q_targets)
    # print(grads)
    # Define the optimizer (e.g., adam)

    """TODOOOO: just add the optimizer???"""


    # model = jax.tree_util.tree_map(lambda m, g: m - LEARNING_RATE * g, model, grads)
    # target_model = jax.tree_util.tree_map(lambda m, g: m - LEARNING_RATE * g, target_model, grads) # FIX TO BE SOFT UPDATE

    # loss, grads = jax.value_and_grad(_batch_loss_fn)(
    #     online_net_params, target_net_params, **experiences
    # )
    # updates, optimizer_state = optimizer.update(grads, optimizer_state)
    # online_net_params = optax.apply_updates(online_net_params, updates)
    # return model, target_model

    # self.optimizer.zero_grad()
    # loss.backward()
    # self.optimizer.step()

    # q_expected_a-q_targets
    print(q_expected)
    print(batch_actions)
    print(q_expected_a)
    # self.qnetwork_local(batch_states).gather(1, batch_actions)
        
    #     ### Loss calculation (we used Mean squared error)
    #     loss = F.mse_loss(q_expected, q_targets)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    # max = 0
    # ind = 0
    # for i in range(BATCH_SIZE):
    #     target_max = jnp.max(target_model(batch[i][3])) 
    #     if jnp.max(target_model(batch[i][3])) > max:
    #         ind = i
    #         max = target_max
    
    # reward = batch[ind][2] + GAMMA * 


    
    
    
            
    
    # print(target_model(jnp.transpose(jnp.array([row[3] for row in batch]))))
    # def learn(self, experiences, gamma):
    #     """Update value parameters using given batch of experience tuples.

    #     Params
    #     ======
    #         experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
    #         gamma (float): discount factor
    #     """
    #     # Obtain random minibatch of tuples from D
    #     states, actions, rewards, next_states, dones = experiences

    #     ## Compute and minimize the loss
    #     ### Extract next maximum estimated value from target network
    #     q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
    #     ### Calculate target value from bellman equation
    #     q_targets = rewards + gamma * q_targets_next * (1 - dones)
    #     ### Calculate expected value from local network
    #     q_expected = self.qnetwork_local(states).gather(1, actions)
        
    #     ### Loss calculation (we used Mean squared error)
    #     loss = F.mse_loss(q_expected, q_targets)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    #     # ------------------- update target network ------------------- #
    #     self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    return 

### ------------- NEURAL NETWORK --------------------------------------------------------



### ------------------------------------------------------------------------------------------
"""
initialize the model and environments

select an action
---- action selected from random with probability, or action selected using
---- argmax thing FIX

exectue action, get observations and see if we stop

add action into REPLAY BUFFER
# target and policy networks >> try to get loss from this
# then optimizer? use optax
# then do gradient descent >> what about actions that are random?

actually need to make sure that its an optimization step 

"""

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

### ------------------------------------------------------------------------------------------

# x_key, y_key, model_key = jax.random.split(jax.random.PRNGKey(0), 3)
# # Example data
# x = jax.random.normal(x_key, (100, 8))
# y = jax.random.normal(y_key, (100, 4))
# model = NeuralNetwork(model_key)

# grads = loss(model, x, y)
# print(grads)
# # Perform gradient descent
# learning_rate = 0.1
# # print(jax.tree_util.tree_map(lambda m, g: m - learning_rate * g, model, grads))

"""

x_2 = jax.random.normal(x_key, (1, 8))
# Compute gradients
# def select_action(obs):
#    # Add in the probability epsilon for choosing specific action
#    # action = env.action_space.sample() 
   
#    for action in 
#    return action

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)


# print(jax.array(observation))
print(model(observation))
print(model(observation))
print(model(observation))

# for _ in range(1000):
#    action = select_action(observation) env.step(action)
#    observation, reward, terminated, truncated, info = 

#    if terminated or truncated:
#       observation, info = env.reset()

# env.close()"""