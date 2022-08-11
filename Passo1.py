# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 21:52:18 2022

@author: MPX
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from math import factorial as fat
from scipy.optimize import differential_evolution
import math
import pandas as pd 
import statistics
import itertools

from scipy.stats import t
from scipy.stats import norm
import matplotlib.pyplot as plt



plt.figure('T-Student e Normal Padrão')
#Gerando a t-student padrão - com v grande e comparando a mesma com a normal
df = 1000 # df é abreveação de degrees of freedom ~ graus de liberdade

mean, var, skew, kurt = t.stats(df, moments='mvsk')

x = np.linspace(t.ppf(0.001, df), t.ppf(0.999, df), 1000)

plt.title('T-Student e Normal Padrão')
mean, var, skew, kurt = norm.stats(10000, moments='mvsk')
plt.plot(x, t.pdf(x, df), '', lw=10, label = 'T-Student')
plt.plot(x, norm.pdf(x), '', lw=4, label = r'Normal ($\mu$ = 0, $\sigma^2$ = 1)', color = 'red')
plt.xlim(-3,3)
plt.grid(visible=True)
plt.legend(title = 'Distribuições')
plt.show()


#______________________________________________________________________________

#Variando a locação
plt.figure(r'Densidade da T-Student - variando $\mu$')
plt.title(r'Densidade da T-Student - variando $\mu$')
#graus de liberdade da t-student
df = 2
locacao = (-5, -1, 0, 3, 5)
for i in locacao:

    mean, var, skew, kurt = t.stats(df, moments='mvsk')
    
    x = np.linspace(i - 5, i + 5, 1000)
    plt.plot(x, t.pdf(x, df, loc = i), '', lw=2, label=rf'$\mu$ = {i}')
    plt.legend()
    plt.grid(visible=True)



plt.show()


#______________________________________________________________________________

#Variando a escala
plt.figure(r'Densidade da T-Student - variando $\sigma^2$')
plt.title(r'Densidade da T-Student - variando $\sigma^2$')
#graus de liberdade da t-student
df = 2
escala = (1, 2, 2.5, 5)
for i in escala:

    mean, var, skew, kurt = t.stats(df, moments='mvsk')
    
    x = np.linspace(-5*i , 5*i, 1000)
    plt.plot(x, t.pdf(x, df, scale = i), '', lw=2, label=rf'$\sigma^2$ = {i}')
    plt.xlim(-10,10)
    plt.legend()
    plt.grid(visible=True)



plt.show()


#______________________________________________________________________________

#Variando os graus de liberdade
plt.figure(r'Densidade da T-Student - variando $\nu$')
plt.title(r'Densidade da T-Student - variando $\nu$')
#graus de liberdade da t-student
df = (2, 5, 10, 30)

for i in df:

    mean, var, skew, kurt = t.stats(i, moments='mvsk')
    
    x = np.linspace(t.ppf(0.001, i), t.ppf(0.999, i), 1000)
    plt.plot(x, t.pdf(x, i), '', lw=2, label=rf'$\nu$ = {i}')
    plt.xlim(-5,5)


plt.plot(x, norm.pdf(x), '', lw=2, label = r'Normal Padrão', color = 'k')
plt.legend()
plt.grid(visible=True)
plt.show()
