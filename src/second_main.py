from vizdoom import *
import random
import time
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from matplotlib import pyplot as plt
import cv2

# intégration de vizdoom dans openai GYM
class VizDoomGym(Env):
    # méthode de démarrage de l'environnement quand on fait une partie.
    def __init__(self, render=False): 
        super().__init__() #hérite de la classe environnement.
        self.game = DoomGame()
        self.game.load_config("github/ViZDoom/scenarios/basic.cfg")

        # On définit si on veut render ou non la game, supprimer le render permettra d'accélérer les calculs en prenant moins de puissance.
        if render == False:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)
        # Initialisation de la game après définition du render
        self.game.init()

        # On définit l'observation de l'environnement.
        self.observation_space = Box(
            low=0, high=255, shape=(100, 160, 1), dtype=np.uint8
        )

        # On définit l'espace d'action
        self.action_space = Discrete(3)

    # comment on fait un "step" dans l'environnement
    def step(self, action):
        # Spécifie les actions possibles et fait une action
        actions = np.identity(3, dtype=np.uint8)
        reward = self.game.make_action(actions[action], 4)

        # Récupère les informations à retourner, gère le cas ou le niveau est fini en retournant une image par défaut et l'info à 0
        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            info = self.game.get_state().game_variables
        else:
            state = np.zeros(self.observation_space.shape)
            info = 0
        done = self.game.is_episode_finished()
        return state, reward, done, info

    # definit comment on render l'environnement ou le jeu
    def render():
        pass

    # définit ce qui se passe quand on fait une nouvelle partie
    def reset(self):
        state = self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.grayscale(state)

    # permet de passer le jeu en echelle de gris et de le resizer pour avoir moins de pixels à processer
    def grayscale(self, observation):
        gray = cv2.cvtColor(
            np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY
        )  # on change l'axe de notre image pour que cv2 focntionne correctement puis on la fait en gris
        resize = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100, 160, 1))
        return state

    # ce qui ferme le jeu
    def close(self):
        self.game.close()
