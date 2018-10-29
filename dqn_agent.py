import bisect
from collections import namedtuple, deque
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

#from model import QNetwork
from model import QNetNorm, QNetDueling, QNetDueling2, QNetNormDueling

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
UPDATE_EVERY = 4       # Number of environment steps between every update with experience replay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Modified from the Udacity Deep Reinforcement Learning Nanodegree course materials.
# Added options for modifying the Q learning, such as double DQN, dueling DQN, and
# prioritized experience replay.
class Agent():
    def __init__(self,
                 state_size,
                 action_size,
                 seed = 0,
                 learning_rate = 1e-3,
                 batch_normalize = False,
                 error_clipping = True,
                 gradient_clipping = False,
                 reward_clipping = False,
                 target_update_interval = -1,
                 double_dqn = True,
                 dueling_dqn = True,
                 prioritized_replay = False):
        """
        state_size (int): Number of parameters in the environment state
        action_size (int): Number of different actions
        seed (int): random seed
        learning_rate (float): initial learning rate
        batch_normlaization (boolean): Flag to use batch normalization in the neural networks.
        error_clipping: Flag for limiting the TD error to between -1 and 1 
        reward_clipping: Flag for limiting the reward to between -1 and 1
        gradient_clipping: Flag for clipping the norm of the gradient to 1 
        target_update_interval: Set negative to use soft updating.
            The number of learning steps between updating the neural network for fixed Q targets.
        double_dqn: Flag for using double Q learning
        dueling_dqn: Flag for using dueling Q networks
        prioritized_replay: Flag for using prioritized replay memory sampling
        
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.learning_rate = learning_rate
                    
        self.error_clipping = error_clipping
        self.reward_clipping = reward_clipping
        self.gradient_clipping = gradient_clipping
        self.target_update_interval = target_update_interval
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn
        self.prioritized_replay = prioritized_replay
        self.target_update_counter = 0
        
        # Q-Network
        if batch_normalize:
            if dueling_dqn:
                self.qnetwork_local = QNetNormDueling(state_size, action_size, seed).to(device)
                self.qnetwork_target = QNetNormDueling(state_size, action_size, seed).to(device)
            else:            
                self.qnetwork_local = QNetNorm(state_size, action_size, seed).to(device)
                self.qnetwork_target = QNetNorm(state_size, action_size, seed).to(device)
        else:
            if dueling_dqn:
                self.qnetwork_local = QNetDueling(state_size, action_size, seed).to(device)
                self.qnetwork_target = QNetDueling(state_size, action_size, seed).to(device)
            else:            
                self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
                self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
            
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr = self.learning_rate)

        # Replay memory
        if self.prioritized_replay:
            self.memory = ReplayBufferPrioritized(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize the time step counter for updating each UPDATE_EVERY number of steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done, gamma = 0.99, beta = 0.0, tau = 0.001):
        
        # reward clipping
        if self.reward_clipping:
            reward = np.clip(reward, -1.0, 1.0)
        
        # Save experience in replay memory
        if self.prioritized_replay:
            if self.double_dqn:
                next_state_index = self.qnetwork_local(torch.FloatTensor(next_state).to(device)).data
                max_next_state_index = torch.argmax(next_state_index)
                next_state_Qvalue = self.qnetwork_target(torch.FloatTensor(next_state).to(device)).data
                max_next_state_Qvalue = next_state_Qvalue[max_next_state_index]
            else:
                next_state_Qvalue = self.qnetwork_target(torch.FloatTensor(next_state)).data
                max_next_state_Qvalue = torch.max(next_state_Qvalue)

            target = reward + gamma * max_next_state_Qvalue * (1 - done)
            old = self.qnetwork_local(torch.FloatTensor(state).to(device)).data[action]
            error = abs(old.item() - target.item())
            
            # TD error clipping
            if self.error_clipping and error > 1:
                error = 1
            
            self.memory.add(state, action, reward, next_state, done, error)
        else:            
            self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get a random subset from the
            # saved experiences (weighted if prior_replay = True) and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, gamma, beta, tau)

    def act(self, state, epsilon=0.0):
        """
        Returns an action for a given state.
        state: current state
        epsilon (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma = 0.99, beta = 0.0, tau = 0.001):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        if self.prioritized_replay:
            states, actions, rewards, next_states, dones, priorities = experiences
        else:
            states, actions, rewards, next_states, dones = experiences
            
        # Get max predicted Q values (for next states) from target model
        if self.double_dqn:
            next_state_local = self.qnetwork_local(next_states).detach()
            max_next_state_indices = torch.max(next_state_local, 1)[1]  
            next_state_Qvalues = self.qnetwork_target(next_states).detach()
            max_next_state_Qvalues = next_state_Qvalues.gather(1, max_next_state_indices.unsqueeze(1))
        else:
            next_state_Qvalues = self.qnetwork_target(next_states).detach()
            max_next_state_Qvalues = torch.max(next_state_Qvalues, 1)[0].unsqueeze(1)

        # Target Q values for current states 
        target_Qvalues = rewards + gamma * max_next_state_Qvalues * (1 - dones)

        # Predicted Q values from local model
        predicted_Qvalues = self.qnetwork_local(states).gather(1, actions)
        
        # Compute errors and loss. Clip the errors so they are between -1 and 1.
        errors = target_Qvalues - predicted_Qvalues
        if self.error_clipping:
            torch.clamp(errors, min=-1, max=1)
        if self.prioritized_replay:
            #beta = 1.0
            Pi = priorities / priorities.sum()
            wi = (1.0 / BUFFER_SIZE / Pi)**beta
            # Normalize wi as per Schaul et al., Prioritized Replay, ICLR 2016, https://arxiv.org/abs/1511.05952
            wi /= max(wi)
            errors *= wi 
        loss = (errors ** 2).mean()

        #loss = F.mse_loss(predicted_Qvalues, target_Qvalues)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1)
        
        self.optimizer.step()

        # Update target network
        if self.target_update_interval >= 0:
            if self.target_update_counter % self.target_update_interval == 0:
                self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
            self.target_update_counter += 1
        else:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, tau)

    # A gradual update of the target network used in the implementation of DQN
    # provided by the Udacity course materials for the deep reinforcement learning nanodegree
    # Basically, a weighted average with a small weight for the local_model
    def soft_update(self, local_model, target_model, tau = 1e-3):
        """
        target_model = τ*local_model + (1 - τ) * target_model

        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau: a small weight for the local model
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

"""
Fixed-size buffer to store experience tuples.
As implemented in the Udacity course materials.
"""
class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        action_size (int): dimension of each action
        buffer_size (int): maximum size of buffer
        batch_size (int): size of each training batch
        seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.memory.append(self.experience(state, action, reward, next_state, done))
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
        
"""
Modified for prioritized experience replay.
Added priority to the experience tuples.
Added weighted random sampling of the experiences.
"""
class ReplayBufferPrioritized:

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha = 0.6, e_priority = 0.01):
        """
        action_size (int): dimension of each action
        buffer_size (int): maximum size of buffer
        batch_size (int): size of each training batch
        seed (int): random seed
        alpha (float): exponent for computing priorities
        e_priority (float): small additive factor to
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "priority"])
        self.seed = random.seed(seed)
        self.alpha = alpha
        self.e_priority = e_priority
    
    def add(self, state, action, reward, next_state, done, error):
        """Add a new experience to memory."""        
        self.memory.append(self.experience(state, action, reward, next_state, done, self.getPriority(error)))
    
    def sample(self):
        """
        Weighted random sampling experiences in memory.
        """        
        priorities = [e.priority for e in self.memory if e is not None]

        experiences = random.choices(population = self.memory, weights = priorities , k = self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        priorities = torch.from_numpy(np.vstack([e.priority for e in experiences if e is not None])).float().to(device)
  
        return (states, actions, rewards, next_states, dones, priorities)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    # Error should be an absolute value of the error.
    # Maybe I should changed this to perform the absolute value here to decrease the likelihood of a mistake
    def getPriority(self, error):
        return(error + self.e_priority) ** self.alpha
    
    
def load_and_run_agent(agent, env, checkpoint, n_episodes):
    '''
    agent: the learning agent
    checkpoint: the saved weights for the pretrained neural network
    n_episodes: the number of episodes to run
    '''
    
    # load the weights from file
    agent.qnetwork_local.load_state_dict(torch.load(checkpoint))
    brain_name = env.brain_names[0]
    
    total_score = 0
    
    for i in range(n_episodes):    
        
        # Reset the environment. Training mode is off.
        env_info = env.reset(train_mode=False)[brain_name]
        
        # Get the current state
        state = env_info.vector_observations[0]
        
        score = 0

        while True:
            # Decide on an action given the current state
            action = agent.act(state)
            
            # Send action to the environment
            env_info = env.step(action.astype(np.int32))[brain_name]
            
            # Get the next state
            next_state = env_info.vector_observations[0]
            
            # Get the reward
            reward = env_info.rewards[0]
            
            # Check if the episode is finished
            done = env_info.local_done[0]
            
            # Add the current reward into the score
            score += reward
            
            state = next_state
            
            # Exit the loop when the episode is done
            if done:
                break
        total_score += score
        
        print("Score: {}".format(score))
        
    print("Average Score: {}".format(total_score / n_episodes))
    
  