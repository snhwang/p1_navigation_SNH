# Deep Reinforcement Learning for Banana Collecting -- Report

This project was one of the requirements for completing the Deep Reinforcement Learning Nanodegree (DRLND) course at Udacity.com. In short, a learning agent is trained to collect yellow bananas (reward of +1) and to avoid blue bananas (reward of -1 if collected). Please refer to README.md for more details about the environment and installation.

## Implementation

The implementation was adapted from the course materials for the coding exercise on deep Q learning. The environment was created using Unity's ML Agents toolkit, but the learning algorithms were implemented with python 3.6 (specifically 3.6.6). The code was run using Jupyter notebook.

Q-learning is a reinforcement learning technique, based on learning a policy which directs the actions taken by an agent given the state of the environment. "Q" is the function which returns the reward for reinforcement and directs the learning of the agent. For deep Q learning, Q is represented by a neural network.

As stated in the README file, the goal was to obtain a score of at least 13 averaged over 100 episodes. This was easily achieved by minimal modification of the code from the coding exercise to train the agent in the Unity ML agent environment rather than an environment from the Open AI Gym. I attempted to approve on this by incorporating variations of deep Q learning that we learned in the course.

### Fixed Q-Targets vs Soft Updating

During Q learning, an error is computed as the difference between the current Q value and the target Q value in order to update the weights. However, the target Q value is not really known and must be estimated. One of the problems with deep Q learning is that the same parameters are used to compute both the current and target Q values and the estimated target values. Since they are correlated and the target is changing, the learning becomes oscillatory.  We learned about the method of fixed Q-targets for reducing this problem.

This required defining 2 networks: qnetwork_local and qnetwork_target. The target network computes estimated targets for updating qnetwork_local. The targets are fixed by only updating the weights in qnetwork_local once every fixed number of steps rather than during every learning step. The interval number of steps can be adjusted as a hyperparameter.

Although we learned about the fixed Q-targets method for reducing this problem, the solution code for the exercise actually used a "soft update" method for gradually blending in the local network to the target network. Basically it is a weighted average, strongly weighted to the original network so that changes occur gradually. The weight for the target network is `tao`, and the weight for the local network is thus `1-tao`. The current code enables the user to choose either the soft update method or provide a target_update_interval which indicates the number of steps between completely replacing the target with the new network.

### Other Options for Reducing Oscillations

#### Limiting Parameters

I implemented options based on other suggestions for possibly reducing oscillations. These are based on limited the range of the parameters which determine the change in weights:

1. Limiting the gradients to between -1 and 1. 

2. Limiting the reward to between -1 and 1.

3. Limiting the magnitude of the time difference error to 1 or less.

For the current environment, the rewards are -1 or 1 for each blue or yellow banana. The limiting of the rewards is thus of minimal use, but I implemented it anyway for future use with other environments. I looked at the actual rewards returned by the environment. Occasionally, I did see rewards of -2 or 2, indicating that occasionally, more than one banana could be picked up following a single action. However, this was uncommon.

I did not see any obvious benefit for these options during my brief testing but would like to test them in a more formal manner. I did not investigate the actual range of gradients and errors experienced during training of this environment.

#### Double Q Learning

In basic Q learning, the same Q function is used to compute both the current Q value as well as for finding the maximum Q value for the target network. Especially in a  noisy environment, this can result in the Q value being overestimated and slowing of the learning process. One partial remedy for this problem is to have 2 independent networks, one for estimating the maximal action and one for computing its Q value. Since we have 2 networks already, qnetwork_local and qnetwork_target, these can be used although they are not entirely independent. The index with the maximum current Q value is determined and then qnetwork_target is used to compute its next state Q value.

### Dueling Q Networks

For dueling Q networks, the network is split into 2 streams, a value function and an advantage function. The Q value, `Q(s, a)`, is a metric of the value of choosing a particular action `a` in a particular state `s`. The value function, `V(s)`, is a metric of the how good it is to be in a particular state `s`. The advantage function  A(s, a) is a metric of the advantage of, or expected improvement to the current policy provided by, a particular action. The Q value is then defined as  V(s) + A(s, a) - mean(s, a). This method is intended to improve the performance of the learning agent. More detals can be found in the following reference:

Wang et al., Dueling Network Architectures for Deep Reinforcement Learning, https://arxiv.org/pdf/1511.06581.pdf

### Experience Replay vs Prioritized Experience Replay

The coding exercise included an implementation of experience replay, which stores prior states and actions. Training is performed on a random sampling of the the stored items. The basic version uniformly samples the items.

In prioritized experience replay, each item is weighted by a function of its error. Since the agent may be able to learn more from the actions which result in a relatively larger error, those with larger errors are weighted more during sampling of the replay memory buffer.

I also varied the size of the memory buffer from 1e3 to 1e6. Zhang and Sutton (A Deeper Look at Experience Replay, https://arxiv.org/abs/1712.01275) suggest that large buffer sizes may hurt learning performance. However, I achieved the best final results with 1e6. The smaller buffer sizes seemed to learn faster in the beginning but leveled off at a lower value.

I also varied the interval from 4 to 16 for updating the neural network with training data sampled from the buffer.  I decided to keep it at 4 because this was the value used in the coding exercise and I did not see any definitive improvement with higher values.

### Evolving Learning Parameters

As is evident, there are many parameters than can be modified to  affect the learning process. In addition, the parameters can be set to change with time since the optimal values may not be static. For example, the original solution code from the coding exercise included an evolving probability, `episilon`, that the agent will choose an action at random. At the start of training, the agent has not learned anything. At this point, a random action should be taken unless there is already some prior knowledge. As the agent learns, `epsilon` should decrease.  I modified the original code to allow `episilon` at time step k to evolve as follows:

`epsilon(k + 1) = epsilon_final + (1 - epsilon_rate) * (epsilon(k) - epsilon_final)`

The following were my final choices for values but I tried different values for the rate:

    epsilon_initial = 1.0,
    epsilon_final = 0.01,
    epsilon_rate = 0.005


Gamma is the discount factor for Q learning for indicating a preference for current rewards over potential future rewards. Francois-Lavet et al. (How to Discount Deep Reinforcement Learning: Towards New Dynamic Strategies, https://arxiv.org/abs/1512.02011) recommend gradually increasing gamma as learning progresses. I use a variation of their method and allow gamma to increase according to:

`gamma(k + 1) = gamma_final + (1 - gamma_rate) * (gamma_final - gamma(k))`

The following were my final choices for values but I tried different values for the rate:

```
gamma_initial = 0.95,
gamma_final = 0.99,
gamma_rate = 0.01
```



Schaul et at. (Prioritized Experience Replay, https://arxiv.org/abs/1511.05952) also varied, beta, the exponential parameter in their importance-sampling weights for prioritized experience replay. I thus also allowed beta to increase as follows:

`beta(k + 1) = 1 + (1 - beta_rate) * (beta(k) - 1)`

The following were my final choices for values but I tried different values for the rate:

```
beta_initial = 0.4,
beta_rate = 0.005
```

Since I had difficulties with the scores leveling off at 16 to 18, depending on my parameters, I also tried allowing tao, the weighting factor for soft updating, to gradually increase:

`tao(k + 1) = tao_final + (1 - tao_rate) * (tao_final - tao(k))`

I set `tao_final` to what I thought we be a fairly high value but made the rate small, so it never reached this value during the number of episodes that I tried. I did it this way since I do not have any intuition for what values would be ideal:

```
tao_initial = 0.001,
tao_final = 0.1,
tao_rate = 0.0001
```



### Neural Networks

The basic neural network, which I obtained from the coding exercise, has the following structure:

1. fully connected layer taking the number of possible states (37) as input and send out 64 outputs
2. Rectified LInear Unit (ReLU)
3. fully connected layer (64 in and out)
4. ReLU
5. fully connected layer(64 in and the number of possible actions out (4))

I also tried a version of dueling Q networks with tanh activation functions but it is currently not an option in the implementation of the learning agent.

Versions with batch normalization have the normalization performed after each ReLU.

The versions for dueling Q Networks both the value and advantage networks share the first fully connected layer and ReLU but then subsequently branch apart to have their own fully connect layers.

The learning utilizes the Adam optimizer.



## Learning Algorithm

#### Agent parameters

The following parameters determine the learning agent:

```
    state_size (int): Number of parameters in the environment state
    action_size (int): Number of different actions
    seed (int): random seed
    learning_rate (float): initial learning rate
    batch_normalize (boolean): Flag for using batch normalization in the neural network
    error_clipping (boolean): Flag for limiting the TD error to between -1 and 1 
    reward_clipping (boolean): Flag for limiting the reward to between -1 and 1
    gradient_clipping (boolean): Flag for clipping the norm of the gradient to 1
    target_update_interval (int): Number of 
    double_dqn (boolean): Flag for using double Q learning
    dueling_dqn (boolean): Flag for using dueling Q networks
    prioritized_replay (boolean): Flag for using prioritized replay memory sampling
```

#### Training parameters

These parameters adjust the learning:

```
agent (Agent): the learning agent
n_episodes (int): the number of epsides for training
epsilon_initial: initial value for the probability that a random action will be chosen
epsilon_final: final epsilon
epsilon_rate = rate metric for decreasing epsilon,
gamma_initial = initial gamma discount factor for the Q learning
gamma_rate = rate metric for increasing gamma up to the the final value (1 - 1 / state_size)
beta_initial(float): Initial exponent for determining the importance-sampling weights for prioritized experience replay
beta_rate (float): Rate metric for increasing beta to 1
```

The training is performed by looping through multiple episodes for the environment.

For each episode

1. Get an action
2. Send the action to the environment
3. Get the next state
4. Get the reward
5. Add the state, action, reward, and next state to the experience replay buffer. Add the error as well for prioritized experience replay
6. Every 4 time steps in the environment, randomly sample the experience replay buffer and perform a learning step (I also tried interval of 8 and 16). If soft updating is performed the weights of the neural network are update every learning step. If fixed Q targets are used, the target neural network weights are only updated at separate intervals specified in the parameters.
7. The reward is added to the score
8. Update the state to the next state and loop back to step 1.



This is implemented in section 3 of the Jupyter notebook, Navigation_SNH.ipynb , as `dqn`. Running `dqn` returns the scores from all of the episodes.



## Results and Plots of Rewards

As I stated, I had very little difficult achieving an average score of 13 but struggled to get much better. My best average scores were just over 18 although individual scores were sometimes in the mid twenties. There was a lot of variability and individual single digit scores  pulled down the averages. I did not perform a systematic exploration of all the parameters although I varied all of them to some degree. After a multiple runs, there was no obvious improvement obtained by limiting parameters (gradients, errors, etc) or with batch normalization, so I set these flags to false. I decided on soft updating rather than fixed Q targets because of better initial results. Double DQN and dueling Q networks appeared to improve the learning obtained by the agent. Prioritized experience replay seemed helpful sometimes when I tried the same parameters without and without it. However, my results were not always consistent and any improvement was at most  minimal. During learning, the initial improvement in scores was variable and more easy to influence than the final scores. I noticed that faster initial improvement did not equate to a better steady state final score. The number of episodes to achieve an average score of 13 varied from 300 to 1200.

The run plotted below used the following parameters:

`agent = Agent(`
​    `state_size=state_size,`
​    `action_size=action_size,`
​    `seed=0,`
​    `batch_normalize = False,`
​    `error_clipping = False,`
​    `reward_clipping = False,`
​    `gradient_clipping = False,`
​    `double_dqn = True,`
​    `prioritized_replay = True,`
​    `dueling_dqn = True,`
​    `learning_rate = 0.0005`
`)`

`scores = dqn(`
​    `agent,`
​    `n_episodes = 5000,`
​    `epsilon_initial = 1.0,`
​    `epsilon_final = 0.01,`
​    `epsilon_rate = 0.005,`
​    `gamma_initial = 0.90,`
​    `gamma_final = 0.99,`
​    `gamma_rate = 0.01,`
​    `beta_initial = 0.4,`
​    `beta_rate = 0.005,`
​    `tau_initial = 0.001,`
​    `tau_final = 0.1,`
​    `tau_rate = 0.00001`
`)`

In addition, hard coded parameters in dqn_agent.py which are not changeable via function parameters include the buffer size of 1e6 for experience replay and an interval of 4 time steps for updating the neural network with data sampled from the the buffer.



![img](data:image/png;base64,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)

The minimum required score of 13 was surpassed at 600 episodes (average score of 13.78 for episodes 501 to 600 during training). Average score over 100 episodes with no training after training for 5000 episodes was 18.62. The checkpoints for this run were renamed to checkpoint13_buffer1e6_every4_tau.pth and checkpoint_final_buffer1e6_every4_tau.pth.

## Ideas for Future Work

The code requires optimization especially the prioritized replay memory implementation.

Although I attempted an implementation for all of the options described. It's still unclear whether I did them correctly since I was not able to achieve significant improvement. This could be related to a coding error or possible the nature of this specific environment. More formal testing is necessary

Other techniques to implement that were mentioned in the course include multi-step bootstrap targets (e.g. A3C), distribution DQN, and noisy DQN, basically the other methods included in the Rainbow algorithm.

