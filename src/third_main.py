from vizdoom import *
import random
import time
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
from matplotlib import pyplot as plt
import cv2

import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import env_checker
from stable_baselines3 import PPO

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
            ammo = self.game.get_state().game_variables[0]
            info = ammo
        else:
            state = np.zeros(self.observation_space.shape)
            info = 0
        info = {"info": info}
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

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self): 
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

CHECKPOINT_DIR = './train/train_basic'
LOG_DIR = 'logs/log_basic'

callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

env = VizDoomGym()
model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps= 2048)
#model.learn(total_timesteps=100000, callback=callback)

#importation de l'évaluateur pour le modèle. 
from stable_baselines3.common.evaluation import evaluate_policy
model = PPO.load('./train/train_basic/best_model_100000')
env = VizDoomGym(render=True)
#mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100)

for episode in range(15): 
    obs = env.reset()
    done = False 
    total_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        time.sleep(0.05)
        total_reward += reward
    print('Total reward pour l épisode {} est {}'.format(episode, total_reward))
    time.sleep(2)