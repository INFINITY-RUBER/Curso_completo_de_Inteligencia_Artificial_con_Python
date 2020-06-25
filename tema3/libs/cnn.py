#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 19:01:47 2018

@author: juangabriel
"""

import torch

class CNN(torch.nn.Module):
    """
    Una red neuronal convolucional que tomará decisiones según los píxeles de la imagen
    """
    def __init__(self, input_shape, output_shape, device = "cpu"):
        """
        :param input_shape: Dimensión de la imagen, que supondremos viene reescalada a Cx84x84
        :param output_shape: Dimensión de la salida
        :param device: El dispositivo (CPU o CUDA) donde la CNN debe almacenar los valores a cada iteración
        """
        #input_shape Cx84x84
        super(CNN, self).__init__() # llamara a la super clase y inicialisar los parametros de la misma
        self.device = device
        
        self.layer1 = torch.nn.Sequential( # transforama a un conjunto de 64 valores posibles
                torch.nn.Conv2d(input_shape[0], 64, kernel_size = 4, stride = 2, padding = 1),
                torch.nn.ReLU() # Función de activación RELU
                )
        
        self.layer2 = torch.nn.Sequential(# transforama a un conjunto de entrada de 64 valores  a uno conjunto de 32 posibles
                torch.nn.Conv2d(64, 32, kernel_size = 4, stride = 2, padding = 0),
                torch.nn.ReLU()
                )
        
        self.layer3 = torch.nn.Sequential(# tomara un entrada de 32 y lo mapeara  a 32 u una ventana mas pequeña de 3 
                torch.nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 0),
                torch.nn.ReLU()
                )# lo que hace es resumir las imagenes y ir quedandonos con los detalles de la imagen
        
        self.out = torch.nn.Linear(18*18*32, output_shape) # 18*18*32 rasgos de entrada  y pasar a lo valores de salida (acciones posibles)
        
    def forward(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.out(x)
        return x
