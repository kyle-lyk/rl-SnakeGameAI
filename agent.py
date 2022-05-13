from matplotlib.pyplot import plot
import torch
import random
import numpy as np
from collections import deque # Double End Queue

from snake_game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from plot import plot

## Game Settings
BLOCK_SIZE = 20

## NN Settings
MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001
I,H,O = 11, 256, 3 ## 11 States, 3 Moves(Straight, Right, Left)

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate (Smaller than 1)
        self.memory = deque(maxlen=MAX_MEMORY) # If memory exceed, popleft()
        self.model = Linear_QNet(I,H,O)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        point_left = Point(head.x-BLOCK_SIZE, head.y)
        point_right = Point(head.x+BLOCK_SIZE, head.y)
        point_up = Point(head.x, head.y-BLOCK_SIZE)
        point_down = Point(head.x, head.y+BLOCK_SIZE)

        direction_left = game.direction == Direction.LEFT
        direction_right = game.direction == Direction.RIGHT
        direction_up = game.direction == Direction.UP
        direction_down = game.direction == Direction.DOWN

        state = [
            # Danger Straight
            (direction_right and game._is_collision(point_right)) or
            (direction_left and game._is_collision(point_left)) or
            (direction_up and game._is_collision(point_up)) or
            (direction_down and game._is_collision(point_down)),

            # Danger Right
            (direction_up and game._is_collision(point_right)) or
            (direction_down and game._is_collision(point_left)) or
            (direction_left and game._is_collision(point_up)) or
            (direction_right and game._is_collision(point_down)),

            # Danger Left
            (direction_up and game._is_collision(point_left)) or
            (direction_down and game._is_collision(point_right)) or
            (direction_right and game._is_collision(point_up)) or
            (direction_left and game._is_collision(point_down)),

            # Move Direction
            direction_left,
            direction_right,
            direction_up,
            direction_down,

            # Food Location
            game.food.x < game.head.x, # Food Left
            game.food.x > game.head.x, # Food Right
            game.food.y < game.head.y, # Food Up
            game.food.y > game.head.y, # Food Down
        ] 

        return np.array(state, dtype=int)

    def cache(self, state, action, reward, next_state, done):
        # popleft is MAX_MEMORY is reached
        self.memory.append((state, action, reward, next_state, done)) # Append only one tuple


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # BATCH SIZE number of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, next_state, done in mini_sample:
        #    self.train_short_memory(state, action, reward, next_state, done)


    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state):
        # Random Moves: Tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games # The more games, the smaller the epsilon (Randomness)
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float32)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item() # EG. [3.23, 2.34, 1.45] -> [1, 0, 0]
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    best_score = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # Get old state
        state_old = agent.get_state(game)

        # Get Move
        final_move = agent.get_action(state_old)

        # Apply Move and Get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train Short memory (Just for previous last move)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Memory
        agent.cache(state_old, final_move, reward, state_new, done)

        if done:
            # Reset the game
            game.reset()
            agent.n_games += 1
            # Train Long Memory (All moves)
            agent.train_long_memory()

            if score > best_score:
                best_score = score
                # Save Best Score
                agent.model.save()

            print('Game: {} Score: {} Best Score: {}'.format(agent.n_games, score, best_score))
            
            total_score += score
            mean_score = total_score / agent.n_games
            # Plot on matplotlib
            plot_on_graph(plot_scores, plot_mean_scores, score, mean_score)


def plot_on_graph(plot_scores, plot_mean_scores, score=0, mean_score=0):
    # Plot on matplotlib
    plot_scores.append(score)
    plot_mean_scores.append(mean_score)
    plot(plot_scores, plot_mean_scores)  

if __name__ == '__main__':
    train()
