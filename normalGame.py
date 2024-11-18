import pygame
from Src.snake_enviroment import SnakeEnv

def play_manual_snake():

    # Initialize the environment
    env = SnakeEnv(grid_size=20)
    
    # Initialize game state
    done = False
    score = 0
    clock = pygame.time.Clock()
    current_action = 1
    
    while not done:

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
                
            if event.type == pygame.KEYDOWN:

                # Handle direction changes
                if event.key == pygame.K_UP and current_action != 1:
                    current_action = 0
                elif event.key == pygame.K_DOWN and current_action != 0:
                    current_action = 1
                elif event.key == pygame.K_LEFT and current_action != 3:
                    current_action = 2
                elif event.key == pygame.K_RIGHT and current_action != 2:
                    current_action = 3
                    
                # Handle speed controls
                elif event.key == pygame.K_SPACE:
                    env.speed_controller.toggle_pause()
                elif event.key == pygame.K_UP and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    env.speed_controller.increase_speed()
                elif event.key == pygame.K_DOWN and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    env.speed_controller.decrease_speed()
        
        # Take action if not paused
        if not env.speed_controller.paused:
            state, reward, done, info = env.step(current_action)
            if reward > 0:
                score += 1
        
        # Render the game
        env.render()
        
        # Control game speed
        if not env.speed_controller.paused:
            clock.tick(10 * env.speed_controller.speed)

    print(f"Game Over! Final Score: {score}")
    pygame.quit()

if __name__ == "__main__":
    play_manual_snake()