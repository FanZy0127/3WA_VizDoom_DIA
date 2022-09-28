import time
import numpy as np
import cv2
# VizDoom
from vizdoom import *
# OpenAI Gym
from gym import Env
from gym.spaces import Discrete, Box
from DoomAi.utils.consts.consts import *


class GymEnv(Env):
    def render(self, mode="human"):
        pass

    def __init__(self,
                 scenario_path,
                 n_actions=N_ACTIONS,
                 damage_taken_delta_weight=DAMAGE_TAKEN_DELTA_WEIGHT,
                 damage_count_delta_weight=DAMAGECOUNT_DELTA_WEIGHT,
                 ammo_delta_weight=AMMO_DELTA_WEIGHT,
                 movement_weight=MOVEMENT_WEIGHT,
                 logging=False,
                 render=False,
                 hd=False):
        # Inherit from Env
        super().__init__()

        # Game Initialization
        self.game = DoomGame()
        self.game.load_config(scenario_path)

        self.logging = logging
        self.n_actions = n_actions
        self.observation_space = Box(low=LOW, high=HIGH, shape=OBSERVATION_SPACE_SHAPE, dtype=np.uint8)
        self.action_space = Discrete(n_actions)

        # Create a vector list represent the actions.
        self.actions = np.identity(n_actions, dtype=np.uint8)

        # Game Variables : HEALTH DAMAGE_TAKEN, DAMAGE_COUNT, SELECTED_WEAPON_AMMO
        self.damage_taken = 0
        self.damage_count = 0
        self.ammo = 52

        # Reward Shaping Weight
        self.movement_weight = movement_weight
        self.damage_taken_delta_weight = damage_taken_delta_weight
        self.damage_count_delta_weight = damage_count_delta_weight
        self.ammo_delta_weight = ammo_delta_weight

        # Render frame configuration
        if not render:
            self.game.set_window_visible(False)

        # High Definition configuration
        if hd:
            self.game.set_screen_resolution(ScreenResolution.RES_800X600)
            self.game.set_render_hud(True)

        self.game.init()

    def step(self, action):
        # Check if the action is legit
        assert action >= 0 & action <= self.n_actions

        # Make an action, then wait 10 frames for loggin
        game_reward = self.game.make_action(self.actions[action])

        if self.logging:
            time.sleep(0.3)

        reward = game_reward

        # Get all the stuff we need to return
        state = self.game.get_state()
        is_done = self.game.is_episode_finished()

        # Check state for prevent error
        if state:
            screen = state.screen_buffer  # Screenshot
            screen = self.grayscale(screen)

            # Reward Shaping
            game_variables = state.game_variables
            health, damage_taken, damage_count, ammo = game_variables

            # Calculate Rewards Delta
            # sd=10dp & d=20dp => -20 + 10 = -10
            # Make understand the agent the need to protect themselves from damage
            damage_taken_delta = -damage_taken + self.damage_taken
            self.damage_taken = damage_taken

            # 0 & 1 = 1 - 0 = 1
            # Make understand the agent that they have to shoot at the targets
            damage_count_delta = damage_count - self.damage_count
            self.damage_count = damage_count

            # 60 & 59 => 59 - 60 => -1
            # Make understand the agent that it is necessary to avoid shooting in the void
            ammo_delta = ammo - self.ammo
            self.ammo = ammo

            movement_reward = game_reward * self.movement_weight
            damage_reward = damage_taken_delta * self.damage_taken_delta_weight
            damage_count_reward = damage_count_delta * self.damage_count_delta_weight
            ammo_reward = ammo_delta * self.ammo_delta_weight

            reward = movement_reward + damage_reward + damage_count_reward + ammo_reward

            if self.logging:
                print('##########################')
                print(f'Action Number: {action}')
                print(f'Health {health}')
                print(f'Ammo Left: {ammo}')
                print(f'Damage Taken: {damage_taken}')
                print(f'Damagecount: {damage_count}')
                print(f'Movement Reward: {movement_reward}')
                print(f'Damage Taken Reward: {damage_reward}')
                print(f'Damagecount Reward: {damage_count_reward}')
                print(f'Ammo Reward: {ammo_reward}')
                print(f'Total Reward: {reward}')
                print('##########################')

            info = {
                "ammo": ammo,
                "health": health,
                "damage_taken": damage_taken,
                "damage_count": damage_count
            }

        else:
            screen = np.zeros(self.observation_space.shape)  # List of 0 with shape of the observation_space
            info = dict()

        return screen, reward, is_done, info

    # Reset the game
    def reset(self):
        # Reset Games Variables before start a new game
        self.damage_taken = 0
        self.damage_count = 0
        self.ammo = 52
        self.game.new_episode()
        screen = self.game.get_state().screen_buffer
        screen = self.grayscale(screen)
        return screen

    # Grayscale the game frame and resize it for better computation time
    @staticmethod
    def grayscale(observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (OBSERVATION_SPACE_SHAPE[1],
                                   OBSERVATION_SPACE_SHAPE[0]), interpolation=cv2.INTER_CUBIC)
        screen = np.reshape(resize, OBSERVATION_SPACE_SHAPE)
        return screen

    # Close the current game session
    def close(self):
        self.game.close()
