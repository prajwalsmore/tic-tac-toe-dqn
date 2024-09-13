# train_tictactoe_agent.py
import torch
import random
from collections import deque
import numpy as np
from tictactoe_env import TicTacToe
from dqn_model import get_model

class Agent:
    def __init__(self, model):
        self.model, self.optimizer, self.criterion = get_model()
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.discount = 0.95
        self.memory = deque(maxlen=2000)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, 8)  # Explore
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()  # Exploit

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target = reward + self.discount * torch.max(self.model(next_state)).item()
            
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state)
            q_target = q_values.clone().detach()
            q_target[0][action] = target

            self.optimizer.zero_grad()
            loss = self.criterion(q_values, q_target)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Training the agent
env = TicTacToe()
agent = Agent(get_model())
episodes = 10000
batch_size = 32

for e in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action, 1)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            print(f"Episode {e+1}/{episodes}, Reward: {total_reward}")
            break

    agent.replay(batch_size)

# Save the model after training
torch.save(agent.model.state_dict(), "tic_tac_toe_dqn.pth")
