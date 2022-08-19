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


'''
O objetivo deste passo é encontrar a máxima verossimilhança de amostras t-student
de tamanho "t_amostra", repetindo o processo "quant_iter" vezes. Para isso, 
utilizaremos a função minimize da biblioteca scipy.optimize e iremos tentar 
reduzir o valor do oposto da soma das logpdf. Com isso, conseguimos recuperar
os valores de mu, sigma^2 e v da t-student original (de acordo com o tamanho
da amostra).
'''


#Definir o tamanho das amostras que serão avaliadas
t_amostra = 50

#Definir a quantidade de vezes que a amostra deverá ser recuperada
quant_iter = 10

#Inserir n = 2 se deseja estimar mu e sigma2. Caso queira estimar o parâmetro v
#defina n com qualquer valor diferente de 2
n = 3

#Definindo o número máximo de iterações a ser considerada na função "minimize"
n_iter = 200


# Gerando uma amostra t-student com locação e escala (diferentes de 0 e 1) 
locacao_base = 10
escala_base = 3
v_base = 3 #df = degrees of freedom, representa o parâmetro (v) da distribuição t




# Função de Máxima Verossimilhança
def MV_t(parametros):
  # Parâmetros a serem calculados
  if n == 2:
      mu, sigma2 = parametros
      v = v_base
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
    chute = (0, 1, 10)
    
#Criando vetores para alocar os valores recuperados e posteriormente calcular
#suas médias

locacao_fit = []
escala_fit = []
v_fit = []
amostra_fit = []

for i in range(quant_iter):
    #Amostragem
    data = t.rvs(v_base, size = t_amostra, loc = locacao_base, scale = escala_base)
    
    # "Minimizando" os parâmetros: função, "chute inicial", método utilizado
    # Método usado no cálculo do erro SLSQP = Programação Sequencial de Min Quadrad.
    mv_modelo = minimize(MV_t, chute, method='SLSQP', options={'maxiter':n_iter}) 
    
    
    #Guardando os valores ajustados pelo modelo de Máxima Verossimilhança 
    amostra_fit.extend(data)
    locacao_fit.append(mv_modelo.x[0])
    escala_fit.append(mv_modelo.x[1])

    if n == 2:
        v_fit.append(v_base)
    else:
        v_fit.append(mv_modelo.x[2])


#Calculando as médias dos valores obtidos

locacao_media = sum(locacao_fit)/float(quant_iter)
escala_media = sum(escala_fit)/float(quant_iter)
v_media = sum(v_fit)/float(quant_iter)

fig, ax = plt.subplots(1, 1)
base = np.linspace(locacao_media - 5*escala_media, locacao_media + 5*escala_media, 1000)
ax.plot(base, t.pdf(base, v_media, loc = locacao_media, scale = escala_media), 'b-', 
        lw=5, alpha=0.6, label='Fit')
ax.plot(base, t.pdf(base, v_base, loc = locacao_base, scale = escala_base), 'r-', 
        lw=5, alpha=0.6, label='Base')

print("\n")
print(rf'A locação da distribuição após o fit foi: u = {locacao_media}')
print(rf'A escala da distribuição após o fit foi: sigma^2 = {escala_media}')
print(rf'O grau de liberdade da distribuição após o fit foi: v = {v_media}')

ax.hist(amostra_fit, bins = 100, density=(True), color = 'skyblue', rwidth=0.9 )
ax.set_xlim(locacao_media - 5*escala_media, locacao_media + 5*escala_media)
ax.legend()
plt.show()
