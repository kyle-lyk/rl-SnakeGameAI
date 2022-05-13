import torch
import torch.nn as nn # Neural Networks
import torch.optim as optim # Optimiser
import torch.nn.functional as F # Activation Function
import os

class Linear_QNet(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super().__init__()
        ## Fully Connected / Feed Forward Layers
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_outputs)

    def forward(self, x): # Prediction
        x = F.relu(self.fc1(x)) # Activation Function
        x = self.fc2(x)
        return x

    def save(self, file_name='linear_qnet.pt'):
        model_path = './model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        file_name = os.path.join(model_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.gamma = gamma
        self.criterion = nn.MSELoss() # Mean Squared Error; Loss = (y - y_pred)^2

    def train_step(self, state, action, reward, next_state, done):
        # Convert to tensor float
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)

        if len(state.shape) == 1: # (1,X) ; # (N,X) = Correct
            # torch.unsqueeze(input, dim=index)
            # Unsqueeze: Add a new dimension to the tensor
            # Eg. (1,X) -> (1,1,X)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: Predicted Q values with current state
        pred = self.model(state)

        # 2: Q_new = reward + gamma * max(next_predicted Q value) -> Only do if not done
        target = pred.clone()
        for i in range (len(done)):
            Q_new = reward[i]
            if not done[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))

            target[i][torch.argmax(action).item()] = Q_new

        # 3: Empty the gradients
        self.optimizer.zero_grad()

        # 4: Calculate loss
        loss = self.criterion(pred, target)

        # 5: Backpropagation
        loss.backward()

        # 6: Update the weights
        self.optimizer.step()
