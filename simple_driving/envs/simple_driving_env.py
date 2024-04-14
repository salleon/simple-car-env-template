import gym
import numpy as np
import math
import pybullet as p
from pybullet_utils import bullet_client as bc
from simple_driving.resources.car import Car
from simple_driving.resources.plane import Plane
from simple_driving.resources.goal import Goal
import time

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class SimpleDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human', 'fp_camera', 'tp_camera']}

    def __init__(self, isDiscrete=True, renders=False):
        if (isDiscrete):
            self.action_space = gym.spaces.Discrete(9)
        else:
            self.action_space = gym.spaces.box.Box(
                low=np.array([-1, -.6], dtype=np.float32),
                high=np.array([1, .6], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-40, -40], dtype=np.float32),
            high=np.array([40, 40], dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()

        if renders:
            self._p = bc.BulletClient(connection_mode=p.GUI)
        else:
            self._p = bc.BulletClient()

        self.reached_goal = False
        self._timeStep = 0.01
        self._actionRepeat = 50
        self._renders = renders
        self._isDiscrete = isDiscrete
        self.car = None
        self.goal_object = None
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()
        self._envStepCounter = 0

        # Q-learning parameters
        self.Q = np.zeros((self.observation_space.shape[0], self.action_space.n))
        self.alpha = 0.1  # learning rate
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.99  # decay rate for epsilon

    def step(self, action):
        # Feed action to the car and get observation of car's state
        if (self._isDiscrete):
            fwd = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
            steerings = [-0.6, 0, 0.6, -0.6, 0, 0.6, -0.6, 0, 0.6]
            throttle = fwd[action]
            steering_angle = steerings[action]
            action = [throttle, steering_angle]
        self.car.apply_action(action)
        for i in range(self._actionRepeat):
            self._p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)

            carpos, _ = self._p.getBasePositionAndOrientation(self.car.car)
            goalpos, _ = self._p.getBasePositionAndOrientation(self.goal_object.goal)
            car_ob = self.getExtendedObservation()

            if self._termination():
                self.done = True
                break
            self._envStepCounter += 1

        # Compute reward as L2 change in distance to goal
        dist_to_goal = np.linalg.norm(carpos - goalpos)
        reward = -dist_to_goal
        self.prev_dist_to_goal = dist_to_goal

        # Update Q-value for previous state and action
        self.Q[self.prev_state, self.prev_action] += self.alpha * (
                    reward + self.gamma * np.max(self.Q[car_ob]) - self.Q[self.prev_state, self.prev_action])

        # Choose next action using epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            action = self.action_space.sample()
        else:
            action = np.argmax(self.Q[car_ob])

        self.prev_state = car_ob
        self.prev_action = action

        ob = car_ob
        return ob, reward, self.done, {}

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._p.resetSimulation()
        self._p.setTimeStep(self._timeStep)
        self._p.setGravity(0, 0, -10)
        # Reload the plane and car
        Plane(self._p)
        self.car = Car(self._p)
        self._envStepCounter = 0

        # Set the goal to a random target
        x = (self.np_random.uniform(5, 9) if self.np_random.integers(2) else
             self.np_random.uniform(-9, -5))
        y = (self.np_random.uniform(5, 9) if self.np_random.integers(2) else
             self.np_random.uniform(-9, -5))
        self.goal = np.array([x, y])
        self.done = False
        self.reached_goal = False

        # Visual element of the goal
        self.goal_object = Goal(self._p, self.goal)

        # Get observation to return
        carpos, _ = self._p.getBasePositionAndOrientation(self.car.car)
        self.prev_state = self.getExtendedObservation()
        self.prev_action = self.action_space.sample()

        self.prev_dist_to_goal = np.linalg.norm(carpos - self.goal)
        car_ob = self.prev_state
        return np.array(car_ob, dtype=np.float32)

    def render(self, mode='human'):
        # Rendering code
        pass

    def getExtendedObservation(self):
        carpos, _ = self._p.getBasePositionAndOrientation(self.car.car)
        goalpos, _ = self._p.getBasePositionAndOrientation(self.goal_object.goal)
        observation = np.array([carpos[0] - goalpos[0], carpos[1] - goalpos[1]], dtype=np.float32)
        return observation

    def _termination(self):
        return self._envStepCounter > 2000

    def close(self):
        self._p.disconnect()
