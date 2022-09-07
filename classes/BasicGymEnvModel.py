import cv2
import numpy as np
# VizDoom
from vizdoom import *
# OpenAI Gym
from gym import Env
from gym.spaces import Discrete, Box
from DoomAi.utils.consts.consts import *


class GymEnv(Env):
    def __init__(self,
                 scenario_path,
                 n_actions=N_ACTIONS,
                 render=False,
                 hd=False):

        super().__init__()

        # Game Initialization
        self.game = DoomGame()
        self.game.load_config(scenario_path)

        self.n_actions = N_ACTIONS
        self.observation_space = Box(low=LOW, high=HIGH, shape=OBSERVATION_SPACE_SHAPE, dtype=np.uint8)
        self.action_space = Discrete(self.n_actions)

        # Transform Actions list to a vector list.
        # Ex: [MOVE_RIGHT, MOVE_LEFT, ATTACK] => [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.actions = np.identity(self.n_actions, dtype=np.uint8)

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

        # Make an action, then wait {{SKIP_FRAMES}} frames
        reward = self.game.make_action(self.actions[action], SKIP_FRAMES)

        # Get all the stuff we need to return
        state = self.game.get_state()
        is_done = self.game.is_episode_finished()

        # Check state for prevent error
        if state:
            screen = state.screen_buffer  # Screenshot
            screen = self.grayscale(screen)
            ammo = state.game_variables[0]  # Ammo Left
            info = {
                "ammo": ammo
            }
        else:
            screen = np.zeros(self.observation_space.shape)  # List of 0 with shape of the observation_space
            info = dict()

        return screen, reward, is_done, info

    # Reset the game
    def reset(self):
        self.game.new_episode()

        screen = self.game.get_state().screen_buffer
        screen = self.grayscale(screen)

        return screen

    # Grayscale the game frame and resize it for better computation time
    @staticmethod
    def grayscale(observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (
            OBSERVATION_SPACE_SHAPE[1], OBSERVATION_SPACE_SHAPE[0]), interpolation=cv2.INTER_CUBIC)
        screen = np.reshape(resize, OBSERVATION_SPACE_SHAPE)
        return screen

    # Close the current game session
    def close(self):
        self.game.close()

    def render(self):
        pass
