import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import random
import numpy as np
from collections import deque
import arcade

# --- Ayarlar ---
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
GRID_SIZE = 20
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
BATCH_SIZE = 64
MEMORY_SIZE = 2000

# --- Aksiyonlar ---
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

# --- DQN Modeli ---
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, len(ACTIONS))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# --- Yılan Ajanı ---
class SnakeAgent:
    def __init__(self, model, rank):
        self.model = model
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.rank = rank

    def get_state(self, snake, food):
        head_x, head_y = snake[0]
        food_x, food_y = food
        state = [
            int(head_y < food_y),  # Aşağıda yiyecek var mı?
            int(head_y > food_y),  # Yukarıda yiyecek var mı?
            int(head_x < food_x),  # Sağda yiyecek var mı?
            int(head_x > food_x),  # Solda yiyecek var mı?
            int((head_x, head_y - GRID_SIZE) in snake),  # Yukarıda beden var mı?
            int((head_x, head_y + GRID_SIZE) in snake),  # Aşağıda beden var mı?
            int((head_x - GRID_SIZE, head_y) in snake),  # Solda beden var mı?
            int((head_x + GRID_SIZE, head_y) in snake)   # Sağda beden var mı?
        ]
        return np.array(state, dtype=int)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(ACTIONS)
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.rank)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return ACTIONS[torch.argmax(q_values).item()]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.rank)
                target = reward + DISCOUNT_FACTOR * torch.max(self.model(next_state_tensor)).item()
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.rank)
            target_f = self.model(state_tensor)
            action_index = ACTIONS.index(action)
            target_f[action_index] = target
            self.optimizer.zero_grad()
            output = self.model(state_tensor)
            loss = self.criterion(output, target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY

# --- Dağıtık Eğitim Fonksiyonu ---
def train(rank, world_size):
    # Dağıtık eğitim için çevresel değişkenleri ayarla
    dist.init_process_group(backend="gloo", init_method="env://", rank=rank, world_size=world_size)
    model = DQN().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    agent = SnakeAgent(ddp_model, rank)

    if rank == 0:
        # Yalnızca ana işçide (rank=0) görsel oyun başlat
        game = SnakeGame(agent)
        arcade.run()  # Arcade döngüsünü başlat

    else:
        # Diğer işçiler sadece model eğitimi yapar
        for episode in range(10):
            snake = [(300, 300), (280, 300), (260, 300)]
            direction = "RIGHT"
            food = (random.randint(0, 29) * GRID_SIZE, random.randint(0, 29) * GRID_SIZE)
            done = False

            while not done:
                state = agent.get_state(snake, food)
                action = agent.choose_action(state)
                direction = update_direction(action, direction)
                snake = move_snake(snake, direction)

                reward = -0.1
                if snake[0] == food:
                    reward = 1
                    food = (random.randint(0, 29) * GRID_SIZE, random.randint(0, 29) * GRID_SIZE)
                    snake.append(snake[-1])

                if (snake[0] in snake[1:] or
                    snake[0][0] < 0 or snake[0][0] >= SCREEN_WIDTH or
                    snake[0][1] < 0 or snake[0][1] >= SCREEN_HEIGHT):
                    reward = -1
                    done = True

                next_state = agent.get_state(snake, food)
                agent.remember(state, action, reward, next_state, done)
                agent.replay()

    dist.destroy_process_group()

# --- Yılan Yön Güncelleme ---
def update_direction(action, direction):
    if action == "UP" and direction != "DOWN":
        return "UP"
    elif action == "DOWN" and direction != "UP":
        return "DOWN"
    elif action == "LEFT" and direction != "RIGHT":
        return "LEFT"
    elif action == "RIGHT" and direction != "LEFT":
        return "RIGHT"
    return direction

# --- Yılan Hareketi ---
def move_snake(snake, direction):
    x, y = snake[0]
    if direction == "UP":
        y += GRID_SIZE
    elif direction == "DOWN":
        y -= GRID_SIZE
    elif direction == "LEFT":
        x -= GRID_SIZE
    elif direction == "RIGHT":
        x += GRID_SIZE
    new_head = (x, y)
    return [new_head] + snake[:-1]

# --- Arcade Oyunu ---
class SnakeGame(arcade.Window):
    def __init__(self, agent):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, "Distributed Snake AI Game")
        self.agent = agent
        self.reset()

    def reset(self):
        self.snake = [(300, 300), (280, 300), (260, 300)]
        self.direction = "RIGHT"
        self.food = self.place_food()

    def place_food(self):
        return (random.randint(0, (SCREEN_WIDTH // GRID_SIZE) - 1) * GRID_SIZE,
                random.randint(0, (SCREEN_HEIGHT // GRID_SIZE) - 1) * GRID_SIZE)

    def on_draw(self):
        arcade.start_render()
        for segment in self.snake:
            arcade.draw_rectangle_filled(segment[0], segment[1], GRID_SIZE, GRID_SIZE, arcade.color.GREEN)
        arcade.draw_rectangle_filled(self.food[0], self.food[1], GRID_SIZE, GRID_SIZE, arcade.color.RED)

    def on_update(self, delta_time):
        state = self.agent.get_state(self.snake, self.food)
        action = self.agent.choose_action(state)
        self.direction = update_direction(action, self.direction)
        self.snake = move_snake(self.snake, self.direction)

        if self.snake[0] == self.food:
            self.food = self.place_food()
            self.snake.append(self.snake[-1])

        if (self.snake[0] in self.snake[1:] or
            self.snake[0][0] < 0 or self.snake[0][0] >= SCREEN_WIDTH or
            self.snake[0][1] < 0 or self.snake[0][1] >= SCREEN_HEIGHT):
            self.reset()

# --- Ana Fonksiyon ---
def main():
    world_size = int(os.environ['WORLD_SIZE'])
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = '192.168.1.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['WORLD_SIZE'] = '4'

    os.environ['RANK'] = '0'  # Bu değer her cihazda uygun şekilde değiştirilmelidir

    main()
