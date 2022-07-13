from gym_game.envs.drone import *
import gym
from gym import spaces
import numpy as np
import random
import pygame
from gym_game.envs.running_env import pygame_events


# the environment must contain 4 functions :__init__,reset,step and render
# we used this to create our environment: https://www.youtube.com/watch?v=ZxXKISVkH6Y

class CustomEnv(gym.Env):
    # This is our Custom environment for 2d drone with rl agent using openAI GYM
    def __init__(self, ):
        # parameters, initial values and defining the screen
        pygame.init()
        self.display = pygame.display.set_mode((800, 800))  # GUI
        self.time = pygame.time.Clock()  # time
        self.line = []  # the line of the flight
        self.create_drone()  # create the drone object
        self.time_step = 0  # the current time step
        self.finished = False  # if the episode is over

        # Creating destination point
        while True:
            self.x_destination = random.uniform(50, 750)
            self.y_destination = random.uniform(50, 750)
            if not ((100 <= self.x_destination <= 130 and 100 <= 800 - self.y_destination <= 130) or (
                    600 <= self.x_destination <= 630 and 100 <= 800 - self.y_destination <= 130) or (
                            500 <= self.x_destination <= 530 and 400 <= 800 - self.y_destination <= 430) or (
                            300 <= self.x_destination <= 330 and 400 <= 800 - self.y_destination <= 430) or (
                            200 <= self.x_destination <= 230 and 700 <= 800 - self.y_destination <= 730) or (
                            700 <= self.x_destination <= 730 and 700 <= 800 - self.y_destination <= 730)):
                break

        # action space is 2 - left motor and right motor
        self.action_space = spaces.Box(low=np.array([-1, -1], dtype=np.float32),
                                       high=np.array([1, 1], dtype=np.float32), dtype=np.float32)

        # observation space is 8 - [speed_x, speed_y, phi, angle, distance_x, distance_y, position_x, position_y]
        self.observation_space = spaces.Box(low=np.array([-1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float32),
                                            high=np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32), dtype=np.float32)

    # create the drone object
    def create_drone(self):
        self.space = pymunk.Space()  # start simulation
        self.space.gravity = Vec2d(0, -1000)  # gravity

        # starting position of the drone
        while True:
            starting_x = random.uniform(200, 600)
            starting_y = random.uniform(200, 600)
            if not ((100 <= starting_x <= 130 and 100 <= 800 - starting_y <= 130) or (
                    600 <= starting_x <= 630 and 100 <= 800 - starting_y <= 130) or (
                            500 <= starting_x <= 530 and 400 <= 800 - starting_y <= 430) or (
                            300 <= starting_x <= 330 and 400 <= 800 - starting_y <= 430) or (
                            200 <= starting_x <= 230 and 700 <= 800 - starting_y <= 730) or (
                            700 <= starting_x <= 730 and 700 <= 800 - starting_y <= 730)):
                break
        starting_angle = random.uniform(-np.pi / 4, np.pi / 4)  # Random starting angle
        self.drone = PyGame2D(starting_x, starting_y, starting_angle, 20, 100, 0.2, 0.4, 0.4,
                              self.space)  # creating drone object

    # an action is preformed
    def step(self, action):

        self.drone.body.body.apply_force_at_local_point(Vec2d(0, (action[0] / 2 + 0.5) * 1000),
                                                        (-self.drone.radius,
                                                         0))  # Same force in the opposite direction to gravity to the left motor
        self.drone.body.body.apply_force_at_local_point(Vec2d(0, (action[1] / 2 + 0.5) * 1000),
                                                        (self.drone.radius,
                                                         0))  # Same force in the opposite direction to gravity to the right motor

        self.space.step(1.0 / 60)  # speed of the simulation
        self.time_step += 1
        self.add_current_position()

        # Calculating reward function
        observation = self.get_observation()
        current_x = observation[1]
        current_y = observation[2]
        observation = observation[0]
        # get 100 reward when reaching the destination, otherwise, get -0.01 for each step
        if np.abs(observation[4]) <= 0.05 and np.abs(observation[5]) <= 0.02:
            reward = 100
            self.finished = True
        else:
            reward = -0.01
        # get -10 reward when out of boundaries or the drone is vertical (meaning the angle of the drone is too big)
        if np.abs(observation[3]) == 1 or np.abs(observation[6]) == 1 or np.abs(observation[7]) == 1:
            reward = -10
            self.finished = True
        # get -10 reward when bump into the red boxes
        if (100 <= current_x <= 130 and 100 <= current_y <= 130) or (
                600 <= current_x <= 630 and 100 <= current_y <= 130) or (
                500 <= current_x <= 530 and 400 <= current_y <= 430) or (
                300 <= current_x <= 330 and 400 <= current_y <= 430) or (
                200 <= current_x <= 230 and 700 <= current_y <= 730) or (
                700 <= current_x <= 730 and 700 <= current_y <= 730):
            reward = -10
            self.finished = True
        # Stops episode, when max time steps is reached
        if self.time_step == 500:
            self.finished = True
        # self.render()
        return observation, reward, self.finished, {}

    # Returns the observation
    def get_observation(self):
        speed_x, speed_y = self.drone.body.body.velocity_at_local_point((0, 0))  # speed of the drone in 2 axis
        speed_x = np.clip(speed_x / 1330, -1, 1)
        speed_y = np.clip(speed_y / 1330, -1, 1)

        phi = self.drone.body.body.angular_velocity  # Angular velocity of the drone
        phi = np.clip(phi / 11.7, -1, 1)

        angle = self.drone.body.body.angle  # The angle of the drone
        angle = np.clip(angle / (np.pi / 2), -1, 1)

        current_x, current_y = self.drone.body.body.position  # Current position

        # If the destination is right or left  from the current position
        if current_x < self.x_destination:
            distance_x = np.clip((current_x / self.x_destination) - 1, -1, 0)
        else:
            distance_x = np.clip(
                (-current_x / (self.x_destination - 800) + self.x_destination / (self.x_destination - 800)), 0,
                1)
        if current_y < self.y_destination:
            distance_y = np.clip((current_y / self.y_destination) - 1, -1, 0)
        else:
            distance_y = np.clip(
                (-current_y / (self.y_destination - 800) + self.y_destination / (self.y_destination - 800)), 0,
                1)
        # current position with normalize
        position_x = np.clip(current_x / 400.0 - 1, -1, 1)
        position_y = np.clip(current_y / 400.0 - 1, -1, 1)

        return np.array([speed_x, speed_y, phi, angle, distance_x, distance_y, position_x, position_y]), current_x, (
                800 - current_y)

    # render the game (show the game)
    def render(self, mode='human', close=False):

        pygame_events(self.space, self)  # from running_env.py
        self.display.fill((255, 255, 255))  # color the window

        # Create the red boxes
        pygame.draw.rect(self.display, (243, 0, 0), pygame.Rect(100, 100, 30, 30))
        pygame.draw.rect(self.display, (243, 0, 0), pygame.Rect(600, 100, 30, 30))
        pygame.draw.rect(self.display, (243, 0, 0), pygame.Rect(500, 400, 30, 30))
        pygame.draw.rect(self.display, (243, 0, 0), pygame.Rect(300, 400, 30, 30))
        pygame.draw.rect(self.display, (243, 0, 0), pygame.Rect(200, 700, 30, 30))
        pygame.draw.rect(self.display, (243, 0, 0), pygame.Rect(700, 700, 30, 30))

        # self.space.debug_draw(self.draw_options)
        pygame.draw.circle(self.display, (255, 0, 0), (self.x_destination, 800 - self.y_destination),
                           5)  # Destination point

        # The line of the flight
        if len(self.line) > 2:
            pygame.draw.aalines(self.display, (0, 0, 0), False, self.line)

        # Put the image of the drone on it's location
        image = pygame.image.load("drone.png")
        image = pygame.transform.scale(image, (80, 10))
        x, y = self.drone.body.body.position
        self.display.blit(pygame.transform.rotate(image, self.drone.body.body.angle * 180.0 / np.pi),
                          pygame.transform.rotate(image, self.drone.body.body.angle * 180.0 / np.pi).get_rect(
                              center=(x, 800 - y)))
        pygame.display.flip()
        self.time.tick(60)

    # Resets the game and returns the first observation data from the game
    def reset(self):
        self.__init__()
        return self.get_observation()[0]

    # End program
    def close(self):
        pygame.quit()

    # Add current position to the list
    def add_current_position(self):
        x, y = self.drone.body.body.position
        self.line.append((x, 800 - y))
