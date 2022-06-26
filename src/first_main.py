from vizdoom import *
import random
import time
import numpy as np
import cv2

#Initialisation de la partie
game = DoomGame()
game.load_config('github/ViZDoom/scenarios/basic.cfg')
game.init()

#Créations des actions 
actions = np.identity(3, dtype=np.uint8)

#loop des épisodes (donc des parties)
episodes = 10 
for episode in range(episodes):
    #creation d'une nouvelle partie
    game.new_episode()
    #on vérifie que la partie n'est pas finie
    while not game.is_episode_finished():
        state = game.get_state()
        img = state.screen_buffer # on recupère létat de l'écran
        print(img.shape)
        info = state.game_variables # on récupère les variables de la game
        reward = game.make_action(random.choice(actions), 4)  # on fait une action random, on skippe 4 frames avant de faire une autre action pour avoir le résultat concret de l'action
        print('reward : ', reward)# on recupere le reward de l'action random
        time.sleep(0.02) # on sleep pour que ce soit visuel
    print('Result', game.get_total_reward()) # on récupère le reward total de la game. 
    time.sleep(2)

game.get_state().screen_buffer.shape