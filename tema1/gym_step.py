#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 18:26:17 2018

@author: juangabriel
"""

import gym 

environment = gym.make("Qbert-v0") 
MAX_NUM_EPISODES = 10 # numero de episodios
MAX_STEPS_PER_EPISODE = 500

for episode in range(MAX_NUM_EPISODES):
    obs = environment.reset() # tomar la primera obsevacion del entorno

    for step in range(MAX_STEPS_PER_EPISODE):
        environment.render() # para pintar en pantalla
        
        action = environment.action_space.sample()## Tomamos una decisión aleatoria...
        next_state, reward, done, info = environment.step(action) # acciones que lleva a cavo y genera 4 resulstados
        obs = next_state  # recupero el nuevo estado del entordon al pasar
        
        if done is True: # si he terminado
            print("\n Episodio #{} terminado en {} steps.".format(episode, step+1))
            break
        
environment.close() # Cerramos la sesión de Open AI Gym