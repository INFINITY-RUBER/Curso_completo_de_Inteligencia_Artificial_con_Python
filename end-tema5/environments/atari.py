#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:20:42 2018

@author: juangabriel
"""

import gym
import atari_py
import numpy as np
import random
import cv2
from collections import deque
from gym.spaces.box import Box

def get_games_list():
    return atari_py.list_games()


def make_env(env_id, env_conf):
    env = gym.make(env_id)
    if 'NoFrameskip' in env_id:
        assert 'NoFrameskip' in env.spec.id
        env = NoopResetEnv(env, noop_max = 30)# toma un primer numero aleatorio para las acciones que vamos a tomar
        env = MaxAndSkipEnv(env, skip = env_conf['skip_rate'])# establese cuantas veses deseas repetir
    
    if env_conf['episodic_life']:
        env = EpisodicLifeEnv(env)# marca un fin de vida en cada episodio del juego terminado
        
    try: 
        if 'FIRE' in env.unwrapped.get_action_meanings(): # si es un de las opciones disponibles
            env = FireResetEnv(env) # sirbe para llevar un accion de disparo en el reset
    except AttributeError:
        pass
    
    env = AtariRescale(env, env_conf['useful_region']) # sirbe para rescalara la imagen del 84 x 84 x 1
    
    if env_conf['normalize_observation']:
        env = NormalizedEnv(env) # para normalizar la imagen observada vasados en la media y varianza observada en la foto actual 
    
        
    env = FrameStack(env, env_conf['num_frames_to_stack'])
    
    if env_conf['clip_reward']: # para evitar valores muy positivos o muy negativos
        env = ClipReward(env)
        
    return env
    
    
def process_frame_84(frame, conf): # para que quede de  84 x 84 el tamaño
    frame = frame[conf["crop1"]:conf["crop2"]+160, :160]
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= 1.0/255.0 # para que quede en escala entre 0 - 1
    frame = cv2.resize(frame, (84, conf["dimension2"]))
    frame = cv2.resize(frame, (84,84))
    frame = np.reshape(frame, [1,84,84])
    return frame

class AtariRescale(gym.ObservationWrapper): # clase que se encarga de hacer el rescalado
    def __init__(self, env, env_conf):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(0, 255, [1,84,84], dtype = np.uint8)
        self.conf = env_conf
        
    def observation(self, observation):
        return process_frame_84(observation, self.conf)

class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env = None):
        gym.ObservationWrapper.__init__(self, env)
        self.mean = 0
        self.std = 0
        self.alpha = 0.9999
        self.num_steps = 0
        
    def observation(self, observation):
        self.num_steps +=1
        self.mean = self.mean * self.alpha + observation.mean() * (1-self.alpha)# media
        self.std = self.std * self.alpha + observation.std() * (1-self.alpha)# desviacion tipica
        
        unbiased_mean = self.mean / (1-pow(self.alpha, self.num_steps)) # genara un media sin sesgo
        unbiased_std = self.std / (1-pow(self.alpha, self.num_steps)) # genara un desviacion sin sesgo
        
        return (observation - unbiased_mean) / (unbiased_std + 1e-8)

class ClipReward(gym.RewardWrapper):# politica de recortar la recompenza
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        
    def reward(self, reward):
        return np.sign(reward)
    

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max = 30):# vamos a tomar un numero aleatoria entre 0 - 30  donde no se va hacer nada 
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP" # se este quieto mientras el entorno se este procesando
        
    def reset(self):
        self.env.reset()
        noops = random.randrange(1, self.noop_max +1)
        assert noops > 0 # solo si es verdadero
        observation = None
        for _ in range(noops): # se saltara las acciones que no generena nada con tra el entorno
            observation, _, done, _ = self.env.step(self.noop_action)
        return observation
    
    def step(self, action):
        return self.env.step(action)
    
class FireResetEnv(gym.Wrapper): # presionara el boton de Fire(disparo)
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3
        
    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs
    
    def step(self, action):
        return self.env.step(action)

class EpisodicLifeEnv(gym.Wrapper): # reseteo segun las vidas de juego
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.has_really_died = False
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.has_really_died = False
        lives = info['ale.lives']
        if lives < self.lives and lives > 0:
            done = True
            self.has_really_died = True
        self.lives = lives
        return obs, reward, done, info
    
    
    def reset(self):
        if self.has_really_died is False:
            obs = self.env.reset()
            self.lives = 0
        else:
            obs,_,_,info = self.env.step(0)
            self.lives = info['ale.lives']
        return obs
    

class MaxAndSkipEnv(gym.Wrapper):# devuelve solo el 4ª freme
    def __init__(self, env = None, skip = 4): # metodo para inicializar
        gym.Wrapper.__init__(self, env)
        self._obs_buffer = deque(maxlen=2)# _obs_buffer es una cola de los que no quiero procesar
        self._skip = skip
        
        
    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis = 0) # me quedo con el maximo valor del buffer
        return max_frame, total_reward, done, info
    
    def reset(self):
        self._obs_buffer.clear() # para limpiar el buffer
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs
    

class FrameStack(gym.Wrapper): # metodo para apilar los frame
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen = k)
        shape = env.observation_space.shape
        self.observation_space = Box(low = 0, high = 255, shape = (shape[0] * k, shape[1], shape[2]), dtype = np.uint8)
        
    def reset(self):# limpiamos y obtenemos la nueva observacion
        obs = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return self.get_obs()


    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self.get_obs(), reward, done, info
    
    
    def get_obs(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))
    
class LazyFrames(object): # se prepara toda la lista de todos los Frames
    def __init__(self, frames):
        self.frames = frames
        self.out = None
        
        
    def _force(self):
        if self.out is None: # si ya ha sido configurado solo lo devuelvo
            self.out = np.concatenate(self.frames, axis = 0)
            self.frames = None
        return self.out
    
    def __array__(self, dtype = None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out
    
    def __len__(self):
        return len(self._force())
    
    def __getitem__(self, i):
        return self._force()[i]