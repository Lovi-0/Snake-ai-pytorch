# 18.11.24

import os
import random
from collections import deque


# External libraries
import torch
import pygame
import numpy as np
import torch.nn as nn
import torch.optim as optim


# Internal utilities
from Src.model._2 import DQN
from .snake_enviroment import SnakeEnv


class SnakeTrainer:
    def __init__(self, input_size=15, output_size=4):
        self.env = SnakeEnv()
        self.model = DQN(input_size, output_size)
        self.target_model = DQN(input_size, output_size)
        
        # Initialize optimizer before loading weights
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Initialize other parameters
        self.memory = deque(maxlen=100000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10
        self.training_step = 0
        
        # Add weights directory path
        self.weights_dir = "model_weights"
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
            
        # Try to load the latest weights
        self.load_latest_weights()
        
        # Update target model after potentially loading weights
        self.target_model.load_state_dict(self.model.state_dict())

    def save_weights(self, episode):
        """
        Save model weights with episode number
        """
        weights_path = os.path.join(self.weights_dir, f"model_weights_ep_{episode}.pth")

        torch.save({
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, weights_path)

    def load_latest_weights(self):
        """
        Load the latest available weights with enhanced error handling
        """
        weights_files = sorted([f for f in os.listdir(self.weights_dir) if f.startswith("model_weights_ep_")])
        
        if weights_files:  # If there are saved weights
            latest_weights = weights_files[-1]
            weights_path = os.path.join(self.weights_dir, latest_weights)
            
            try:
                checkpoint = torch.load(weights_path, weights_only=True)  # Add weights_only flag

                # Load model state with error handling
                try:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                except RuntimeError as e:
                    print(f"Warning: Could not load model weights due to architecture mismatch: {e}")
                    return False

                # Load optimizer state with error handling
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except ValueError as e:
                    print(f"Warning: Could not load optimizer state: {e}")
                    # Continue even if optimizer state fails to load

                # Load training parameters
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                self.training_step = checkpoint.get('training_step', 0)

                print(f"Loaded weights from episode {checkpoint.get('episode', 'unknown')}")
                return True
            
            except Exception as e:
                print(f"Error loading weights: {e}")
                return False
            
        return False

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(4)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state)
            return q_values.argmax().item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Use numpy for efficient conversion
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to numpy arrays first, then to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)

        # Optional: Add device handling if using GPU
        if torch.cuda.is_available():
            states = states.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            next_states = next_states.cuda()
            dones = dones.cuda()
            self.model = self.model.cuda()
            self.target_model = self.target_model.cuda()

        # Compute Q-values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.training_step += 1

        # Update target network periodically
        if self.training_step % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def train(self, episodes=5000):
        scores = []
        max_score = 0
        max_real_score = 0

        for episode in range(episodes):
            state = self.env.reset()
            score = 0
            done = False

            while not done:

                # Handle events including speed control
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:

                        # Save weights before closing
                        self.save_weights(episode)
                        self.env.close()
                        return scores
                    
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP:
                            self.env.speed_controller.increase_speed()
                        elif event.key == pygame.K_DOWN:
                            self.env.speed_controller.decrease_speed()
                        elif event.key == pygame.K_SPACE:
                            self.env.speed_controller.toggle_pause()

                # Only proceed if not paused
                if not self.env.speed_controller.paused:
                    action = self.act(state)
                    next_state, reward, done, _ = self.env.step(action)
                    self.remember(state, action, reward, next_state, done)
                    self.replay()
                    state = next_state
                    score += reward

                self.env.render()

                # Control speed
                pygame.time.Clock().tick(self.env.speed_controller.speed)

            scores.append(score)
            if score > max_score:
                max_score = score
            
            # Save weights every 100 episodes
            if (episode + 1) % 100 == 0:
                self.save_weights(episode + 1)
                print(f"Saved weights at episode {episode + 1}")

            if self.env._get_score() > max_real_score:
                max_real_score = self.env._get_score()
            
            if episode % 10 == 0:
                print(f"Episode: {episode + 1}, Score: {score:.1f}, Max Score: {max_score:.1f}, Max game score: {max_real_score}, Epsilon: {self.epsilon:.2f}")

        # Save final weights
        self.save_weights(episodes)
        self.env.close()
        return scores
