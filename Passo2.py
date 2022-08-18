# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 18:29:45 2022

@author: MPX
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels import api
from scipy import stats
from scipy.stats import t
from scipy.optimize import minimize 

#Inserir n = 2 se deseja estimar mu e sigma2. Caso queira estimar o parâmetro v
#defina n com qualquer valor diferente de 2
n = 3

#Definindo o número máximo de iterações a ser considerada na função "minimize"
n_iter = 30


# Gerando uma amostra t-student com locação e escala (diferentes de 0 e 1) 
locacao = 10
escala = 3
df = 3 #df = degrees of freedom, representa o parâmetro (v) da distribuição t

#Gerando a amostra que será utilizada para ajustar o modelo posteriormente
data = t.rvs(df, size=500, loc = locacao, scale = escala)
# print(data)


# Função de Máxima Verossimilhança
def MV_t(parametros):
  # Parâmetros a serem calculados
  if n == 2:
      mu, sigma2 = parametros
      v = df
  else: 
      mu, sigma2, v = parametros

  # Calculando a log-máxima verossimilhança seguindo uma distribuição t-student
  LMV = np.sum(stats.t.logpdf(x = data, loc = mu, scale = sigma2, df = v))
  # Inversão do sinal da log-mmax verossimilhança (a fim de minimizar)
  neg_LMV = -1*LMV
  return neg_LMV


#Gerando o chute inicial
if n == 2:
    chute = (0, 1)
else:
    chute = (0, 1, 15)
    
# "Minimizando" os parâmetros: função, "chute inicial", método utilizado
# Método usado no cálculo do erro SLSQP = Programação Sequencial de Min Quadrad.
mv_modelo = minimize(MV_t, chute, method='SLSQP', options={'maxiter':n_iter}) 


#Printando sumário do ajuste
print(mv_modelo)

#Guardando os valores ajustados pelo modelo de Máxima Verossimilhança 
locacao_fit = mv_modelo.x[0]
escala_fit = mv_modelo.x[1]

if n == 2:
    v_fit = df
else:
    v_fit = mv_modelo.x[2]


fig, ax = plt.subplots(1, 1)
base = np.linspace(locacao_fit - 5*escala_fit, locacao_fit + 5*escala_fit, 1000)
ax.plot(base, t.pdf(base, v_fit, loc = locacao_fit, scale = escala_fit), 'r-', 
        lw=5, alpha=0.6, label='t pdf')

print("\n")
print(rf'A locação da distribuição após o fit foi: u = {locacao_fit}')
print(rf'A escala da distribuição após o fit foi: sigma^2 = {escala_fit}')
print(rf'O grau de liberdade da distribuição após o fit foi: v = {v_fit}')

ax.hist(data, 100, density=(True) ,rwidth=0.9)
ax.set_xlim(locacao_fit - 5*escala_fit, locacao_fit + 5*escala_fit)
ax.legend()
plt.show()