# 18.11.24

import random


# External libraries
import gym
import heapq
import pygame
import numpy as np
from gym import spaces


# Internal utilities
from .util.speed import SpeedController


# Variable
classic_render = False


class SnakeEnv(gym.Env):
    def __init__(self, grid_size=20):
        super(SnakeEnv, self).__init__()
        self.grid_size = grid_size
        self.window_size = (800, 600)
        self.speed_controller = SpeedController()
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(15,), dtype=np.float32)
        
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Snake RL Training")
        self.font = pygame.font.Font(None, 36)
        
        self.reset()

    def _find_path_to_apple(self):
        start = self.snake_body[0]
        goal = self.apple
        
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        heap = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while heap:
            current = heapq.heappop(heap)[1]
            
            if current == goal:
                break
                
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                next_pos = (current[0] + dx, current[1] + dy)
                
                if (0 <= next_pos[0] < self.grid_size and 
                    0 <= next_pos[1] < self.grid_size and 
                    next_pos not in self.snake_body):
                    
                    new_cost = cost_so_far[current] + 1
                    if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                        cost_so_far[next_pos] = new_cost
                        priority = new_cost + heuristic(goal, next_pos)
                        heapq.heappush(heap, (priority, next_pos))
                        came_from[next_pos] = current
        
        if goal not in came_from:
            return None
            
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        
        return path
    
    def _place_apple(self):
        while True:
            apple = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            if apple not in self.snake_body:
                return apple

    def _get_state(self):
        head = self.snake_body[0]
        
        # Basic state information
        danger_straight = self._is_danger(self._get_position_in_direction(head, self.snake_direction))
        danger_right = self._is_danger(self._get_position_in_direction(head, self._turn_right(self.snake_direction)))
        danger_left = self._is_danger(self._get_position_in_direction(head, self._turn_left(self.snake_direction)))
        
        dir_up = self.snake_direction == (0, -1)
        dir_down = self.snake_direction == (0, 1)
        dir_left = self.snake_direction == (-1, 0)
        dir_right = self.snake_direction == (1, 0)
        
        apple_left = self.apple[0] < head[0]
        apple_right = self.apple[0] > head[0]
        apple_up = self.apple[1] < head[1]
        apple_down = self.apple[1] > head[1]

        # Path information
        path_exists = self.path_to_apple is not None
        next_in_path = (0, 0, 0, 0)  # up, right, down, left
        if path_exists and len(self.path_to_apple) > 1:
            next_step = self.path_to_apple[1]
            direction_to_next = (next_step[0] - head[0], next_step[1] - head[1])
            next_in_path = (
                direction_to_next == (0, -1),  # up
                direction_to_next == (1, 0),   # right
                direction_to_next == (0, 1),   # down
                direction_to_next == (-1, 0)   # left
            )

        state = np.array([
            danger_straight,
            danger_right,
            danger_left,
            dir_up,
            dir_down,
            dir_left,
            dir_right,
            apple_left,
            apple_right,
            apple_up,
            apple_down,
            *next_in_path
        ], dtype=np.float32)

        return state

    def _is_danger(self, position):
        x, y = position
        return (x < 0 or x >= self.grid_size or 
                y < 0 or y >= self.grid_size or 
                position in self.snake_body)

    def _get_position_in_direction(self, position, direction):
        return (position[0] + direction[0], position[1] + direction[1])

    def _turn_right(self, direction):
        return (direction[1], -direction[0])

    def _turn_left(self, direction):
        return (-direction[1], direction[0])

    def _is_self_collision(self, position):
        # Exclude the last body segment and last head (which will move in the next frame) DIO -.-.
        collision_body = self.snake_body[2:]
        return position in collision_body

    def _is_wall_collision(self, position):
        x, y = position
        return x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size

    def _is_danger(self, position):
        return self._is_wall_collision(position) or position in self.snake_body
    
    def _get_score(self):
        return self.score
    
    def step(self, action):
        self.steps += 1
        
        # Convert action to direction
        if action == 0:  # Up
            self.snake_direction = (0, -1)
        elif action == 1:  # Down
            self.snake_direction = (0, 1)
        elif action == 2:  # Left
            self.snake_direction = (-1, 0)
        elif action == 3:  # Right
            self.snake_direction = (1, 0)

        # Move snake
        new_head = self._get_position_in_direction(self.snake_body[0], self.snake_direction)
        
        # Check different types of collisions
        wall_collision = self._is_wall_collision(new_head)
        self_collision = self._is_self_collision(new_head)
        
        # Game over conditions with different penalties
        if wall_collision or self_collision:
            if self_collision:
                reward = -100
            else:
                reward = -100

            #print("wall_collision: ", wall_collision, " self_collision: ", self_collision, "\b")
            return self._get_state(), reward, True, {'collision_type': 'self' if self_collision else 'wall'}

        self.snake_body.insert(0, new_head)
        
        # Check if apple eaten
        reward = 0
        if new_head == self.apple:
            self.score += 1
            reward = 10
            self.apple = self._place_apple()
            self.path_to_apple = self._find_path_to_apple()

        else:
            self.snake_body.pop()
            
            # Path-based reward
            if self.path_to_apple and len(self.path_to_apple) > 1:
                if new_head == self.path_to_apple[1]:
                    reward = 0.1
                else:
                    reward = -0.1
            
            else:
                reward = -0.1
                
            self.path_to_apple = self._find_path_to_apple()

        # Check if path to apple exist
        try:
            if len(self.path_to_apple) == 0:
                reward = -150
        except:
            reward = -150

        # Additional reward based on distance to apple
        head = self.snake_body[0]
        distance_to_apple = abs(head[0] - self.apple[0]) + abs(head[1] - self.apple[1])
        reward += 0.1 * (1.0 / (distance_to_apple + 1))  # Small reward for getting closer to apple

        return self._get_state(), reward, False, {}

    def render(self):

        self.screen.fill((0, 0, 0))
        
        # Draw grid
        for x in range(0, self.window_size[0], self.window_size[0] // self.grid_size):
            pygame.draw.line(self.screen, (40, 40, 40), (x, 0), (x, self.window_size[1]))  # Vertical lines
        for y in range(0, self.window_size[1], self.window_size[1] // self.grid_size):
            pygame.draw.line(self.screen, (40, 40, 40), (0, y), (self.window_size[0], y))  # Horizontal lines
        

        if classic_render:

            # Draw path
            if self.path_to_apple:
                for pos in self.path_to_apple:
                    pygame.draw.rect(self.screen, (50, 50, 50),
                                (pos[0] * self.window_size[0]/self.grid_size,
                                    pos[1] * self.window_size[1]/self.grid_size,
                                    self.window_size[0]/self.grid_size,
                                    self.window_size[1]/self.grid_size))
            
            # Draw snake
            for idx, segment in enumerate(self.snake_body):
                if idx == 0: color = (0, 0, 255)  # Blue for the head
                else: color = (0, 255, 0)  # Green for the body

                pygame.draw.rect(self.screen, color,
                            (segment[0] * self.window_size[0]/self.grid_size,
                                segment[1] * self.window_size[1]/self.grid_size,
                                self.window_size[0]/self.grid_size,
                                self.window_size[1]/self.grid_size))
            
            # Draw apple
            pygame.draw.rect(self.screen, (255, 0, 0),
                            (self.apple[0] * self.window_size[0]/self.grid_size,
                            self.apple[1] * self.window_size[1]/self.grid_size,
                            self.window_size[0]/self.grid_size,
                            self.window_size[1]/self.grid_size))
            
            # Draw speed info
            #speed_text = self.font.render(f"Speed: {self.speed_controller.speed}x", True, (255, 255, 255))
            #pause_text = self.font.render("PAUSED" if self.speed_controller.paused else "", True, (255, 255, 255))
            score_text = self.font.render(f"S: {self.score}", True, (255, 255, 255))
            
            #self.screen.blit(speed_text, (10, 10))
            #self.screen.blit(pause_text, (10, 50))
            self.screen.blit(score_text, (0, 0))
                
        else:

        
            # Draw path
            if self.path_to_apple:
                for pos in self.path_to_apple:
                    pygame.draw.rect(self.screen, (50, 50, 50),
                                    (pos[0] * self.window_size[0] / self.grid_size,
                                    pos[1] * self.window_size[1] / self.grid_size,
                                    self.window_size[0] / self.grid_size,
                                    self.window_size[1] / self.grid_size))
            
            # Draw snake
            for idx, segment in enumerate(self.snake_body):
                # Determine color: blue for head, green for the rest
                if idx == 0:
                    color = (0, 0, 255)  # Blue for the head
                else:
                    color = (0, 255, 0)  # Green for the body
                
                # Draw the snake segment
                pygame.draw.rect(self.screen, color,
                                (segment[0] * self.window_size[0] / self.grid_size,
                                segment[1] * self.window_size[1] / self.grid_size,
                                self.window_size[0] / self.grid_size,
                                self.window_size[1] / self.grid_size))
                
                # Render the segment number
                segment_text = self.font.render(str(idx), True, (255, 255, 255))  # White numbers
                text_rect = segment_text.get_rect(center=(
                    segment[0] * self.window_size[0] / self.grid_size + self.window_size[0] / (2 * self.grid_size),
                    segment[1] * self.window_size[1] / self.grid_size + self.window_size[1] / (2 * self.grid_size)
                ))
                self.screen.blit(segment_text, text_rect)
            
            # Draw apple
            pygame.draw.rect(self.screen, (255, 0, 0),
                            (self.apple[0] * self.window_size[0] / self.grid_size,
                            self.apple[1] * self.window_size[1] / self.grid_size,
                            self.window_size[0] / self.grid_size,
                            self.window_size[1] / self.grid_size))
            
            # Draw speed info
            speed_text = self.font.render(f"Speed: {self.speed_controller.speed}x", True, (255, 255, 255))
            pause_text = self.font.render("PAUSED" if self.speed_controller.paused else "", True, (255, 255, 255))
            score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
            
            self.screen.blit(speed_text, (10, 10))
            self.screen.blit(pause_text, (10, 50))
            self.screen.blit(score_text, (10, 90))
            
            # Draw controls info
            controls_text = self.font.render("UP/DOWN: Speed | SPACE: Pause", True, (200, 200, 200))
            self.screen.blit(controls_text, (self.window_size[0] - 350, 10))
            
        pygame.display.flip()

    def reset(self):
        self.snake_body = [(self.grid_size//2, self.grid_size//2)]
        self.snake_direction = (0, 1)
        self.apple = self._place_apple()
        self.score = 0
        self.steps = 0
        self.path_to_apple = self._find_path_to_apple()
        return self._get_state()
    