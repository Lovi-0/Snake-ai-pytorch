# 18.11.24

# Variable
MIN_SPEED = 5
MAX_SPEED = 5000
DEFAULT_SPEED = 5000


class SpeedController:
    def __init__(self, initial_speed=DEFAULT_SPEED):
        self.speed = initial_speed
        self.paused = False
        
    def increase_speed(self):
        self.speed = min(MAX_SPEED, self.speed + 5)
        return self.speed
        
    def decrease_speed(self):
        self.speed = max(MIN_SPEED, self.speed - 5)
        return self.speed
        
    def toggle_pause(self):
        self.paused = not self.paused
        return self.paused
