import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Taken from the Udacity course materials.
3 fully connected layers with relu activations.
My versions for dueling DQN, using tanh activation, and batch normalization are below this.
'''
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """
        state_size (int): Number of parameters in the state
        action_size (int): Number of possible actions
        seed (int): Random seed
        fc1_units (int): Number of nodes in first hidden layer
        fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        '''
        Maps state of the environment to actions
        '''
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

'''
Added batch normalization. It didn't seem to help much
'''
class QNetNorm(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        '''
        state_size (int): Number of parameters in the state
        action_size (int): Number of possible actions
        seed (int): Random seed
        fc1_units (int): Number of nodes in first hidden layer
        fc2_units (int): Number of nodes in second hidden layer
        '''
        super(QNetNorm, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)

    def forward(self, state):
        x = self.bn1(F.relu(self.fc1(state)))
        x = self.bn2(F.relu(self.fc2(x)))
        return self.fc3(x)

'''
Modified into a dueling Q network
'''
class QNetDueling(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        '''
        state_size (int): Number of parameters in the state
        action_size (int): Number of possible actions
        seed (int): Random seed
        fc1_units (int): Number of nodes in first hidden layer
        fc2_units (int): Number of nodes in second hidden layer
        '''
        super(QNetDueling, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        self.fc2_state_values = nn.Linear(fc1_units, fc2_units)
        self.fc3_state_values = nn.Linear(fc2_units, 1)
        
        self.fc2_advantage_values = nn.Linear(fc1_units, fc2_units)
        self.fc3_advantage_values = nn.Linear(fc2_units, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))

        x_advantage_values = F.relu(self.fc2_advantage_values(x))
        x_advantage_values = self.fc3_advantage_values(x_advantage_values)
        
        x_state_values = F.relu(self.fc2_state_values(x))
        x_state_values = self.fc3_state_values(x_state_values)
            
        return x_state_values + x_advantage_values - x_advantage_values.mean()
    
'''
Uses tanh activation
'''
class QNetDueling2(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        '''128
        state_size (int): Number of parameters in the state
        action_size (int): Number of possible actions
        seed (int): Random seed
        fc1_units (int): Number of nodes in first hidden layer
        fc2_units (int): Number of nodes in second hidden layer
        '''
        super(QNetDueling2, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        self.fc2_state_values = nn.Linear(fc1_units, fc2_units)
        self.fc3_state_values = nn.Linear(fc2_units, 1)
        
        self.fc2_advantage_values = nn.Linear(fc1_units, fc2_units)
        self.fc3_advantage_values = nn.Linear(fc2_units, action_size)
        
    def forward(self, state):
        x = torch.tanh(self.fc1(state))

        x_state_values = torch.tanh(self.fc2_state_values(x))
        x_state_values = self.fc3_state_values(x_state_values)

        x_advantage_values = torch.tanh(self.fc2_advantage_values(x))
        x_advantage_values = self.fc3_advantage_values(x_advantage_values)
            
        return x_state_values + x_advantage_values - x_advantage_values.mean()
    
'''
Deuling Q Network with batch normalization.
'''
class QNetNormDueling(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetNormDueling, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)

        self.fc2_state_values = nn.Linear(fc1_units, fc2_units)
        self.fc3_state_values = nn.Linear(fc2_units, 1)
        
        self.fc2_advantage_values = nn.Linear(fc1_units, fc2_units)
        self.fc3_advantage_values = nn.Linear(fc2_units, action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.bn1(F.relu(self.fc1(state)))

        x_advantage_values = self.bn2(F.relu(self.fc2_advantage_values(x)))
        x_advantage_values = self.fc3_advantage_values(x_advantage_values)
        
        x_state_values = self.bn2(F.relu(self.fc2_state_values(x)))
        x_state_values = self.fc3_state_values(x_state_values)
            
        return x_state_values + x_advantage_values - x_advantage_values.mean()
    
