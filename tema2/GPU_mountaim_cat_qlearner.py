#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 13:15:19 2018

@author: Ruber hernandez
"""

import gym # importamos la libreria
import numpy as np
import timeit
import tensorflow as tf

# sudo apt-get install ffmpeg
#!pip install ffmpeg  .... libreria para guardado de videos

# EPISILON_MIN : vamos aprendiendo, mientras el incremento de aprendizaje sea superior a dicho valor
# MAX_NUM_EPISONES : número máximo de iteraciones que estamos dispuestos a realizar
# STEPS_PER_EPISODE: número máximo de pasos a realizar en cada episodio
# ALPHA: ratio de aprendizaje del agente -
# GAMMA: factor de descuento del agente (lo que se pierde de un paso a otro)
# NUM_DISCRETE_BINS: número de divisiones en el caso de discretizar el espacio de estados continuo. 

MAX_NUM_EPISODES = 40000
STEPS_PER_EPISODE = 200
EPSILON_MIN = 0.005
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
EPSILON_DECAY = 500 * EPSILON_MIN / max_num_steps
ALPHA = 0.05
GAMMA = 0.98
NUM_DISCRETE_BINS = 30

# QLearner Class
# __init__(self, environment)
# discretize(self, obs) [-2,2] -> [-2,-1], [-1,0], [0,1], [1,2]
# get_action(self, obs)
# learn(self, obs, action, reward, next_obs)

class QLearner(object):
    def __init__(self, environment):  # metodo de inicializacion y self es una referencia del propio objecto
        self.obs_shape = environment.observation_space.shape # me quedo los valores (tamaño, mas alto y mas bajo)
        self.obs_high = environment.observation_space.high
        self.obs_low = environment.observation_space.low
        self.obs_bins = NUM_DISCRETE_BINS  # le paso el nuemero de diviciones 
        self.bin_width = (self.obs_high-self.obs_low)/self.obs_bins # me da el tamaño de ancho de observaciones

        self.action_shape = environment.action_space.n # numero de acciones
        self.Q = np.zeros((self.obs_bins+1, self.obs_bins+1, self.action_shape)) # matriz de 31 x 31 x 3 # Me servira para guarda los estados que por donde va pasando el estado
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = 1.0 # el valor para que vaya decrementado
    
    def discretize(self, obs):
        return tuple(((obs-self.obs_low)/self.bin_width).astype(int)) # operacion para saber en que cuadro esta

    def get_action(self, obs):
        discrete_obs = self.discretize(obs)
        # Selección de la acción en base a Epsilon-Greedy
        if self.epsilon > EPSILON_MIN:
            self.epsilon -= EPSILON_DECAY #que parte de Epsilo voy perdineo en cada paso
        if np.random.random() > self.epsilon: # Con probabilidad 1-epsilon, elegimos la mejor posible
            return np.argmax(self.Q[discrete_obs])
        else:
            return np.random.choice([a for a in range(self.action_shape)])# Con probabilidad epsilon, elegimos una al azar

    def learn(self, obs, action, reward, next_obs):
        discrete_obs = self.discretize(obs)
        discrete_next_obs = self.discretize(next_obs)
        #
        # IMPLEMENTO LA ECUACION DE (Bellman)
        self.Q[discrete_obs][action] += self.alpha*(reward + self.gamma * np.max(self.Q[discrete_next_obs]) - self.Q[discrete_obs][action])


## Método para entrenar a nuestro agente
def train(agent, environment):
    best_reward = -float('inf') # La mejor recompensa
    for episode in range(MAX_NUM_EPISODES):
        done = False
        obs = environment.reset()
        total_reward = 0.0
        while not done:
            action = agent.get_action(obs)# Acción elegida según la ecuación de Q-LEarning #la mejor accion
            next_obs, reward, done, info = environment.step(action)
            agent.learn(obs, action, reward, next_obs) # que aprenda a base de la observacion
            obs = next_obs
            total_reward += reward
        if total_reward > best_reward:
            best_reward = total_reward

        if (episode)% 500 == 0:
            print("EPisodio número {} con recompensa: {}, mejor recompensa: {}, epsilon: {}".format(episode, total_reward, best_reward, agent.epsilon))

    ## De todas las políticas de entrenamiento que hemos obtenido devolvemos la mejor de todas
    return np.argmax(agent.Q, axis = 2)


def test(agent, environment, policy):    
    done = False
    obs = environment.reset()
    total_reward = 0.0
    while not done:
        action = policy[agent.discretize(obs)] #acción que dictamina la política que hemos entrenado 
        next_obs, reward, done, info = environment.step(action)
        obs = next_obs
        total_reward += reward
    return total_reward # retorna la recompesa global


def ejecucion():
    environment = gym.make("MountainCar-v0")
    agent = QLearner(environment)
    learned_policy = train(agent, environment)
    monitor_path = "./monitor_output"
    environment = gym.wrappers.Monitor(environment, monitor_path, force = True)
    for _ in range(1000):
        test(agent, environment, learned_policy) 

if __name__ == "__main__":
    with tf.device('/device:GPU:0'):
        print('GPU_A EJECUTAR: ',tf.test.gpu_device_name())
        ejecucion()
        print('ejecucion finalizada.............')




    