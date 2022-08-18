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
import statistics as st


'''
O objetivo deste passo é encontrar a máxima verossimilhança de amostras t-student
de tamanho "t_amostra", repetindo o processo "quant_iter" vezes. Para isso, 
utilizaremos a função minimize da biblioteca scipy.optimize e iremos tentar 
reduzir o valor do oposto da soma das logpdf. Com isso, conseguimos recuperar
os valores de mu, sigma^2 e v da t-student original (de acordo com o tamanho
da amostra).
'''

#Definindo o erro para a parada do algoritmo EM
erro = 1e-4

#Criando a variável que receberá a norma dos parâmetros
norm = 1 

#Definir o tamanho das amostras que serão avaliadas
t_amostra = 150

#Definir a quantidade de vezes que a amostra deverá ser recuperada
quant_iter = 20

#Inserir n = 2 se deseja estimar mu e sigma2. Caso queira estimar o parâmetro v
#defina n com qualquer valor diferente de 2
n = 2

#Definindo o número máximo de iterações a ser considerada na função "minimize"
n_iter = 200


# Gerando uma amostra t-student com locação e escala (diferentes de 0 e 1) 
locacao_base = 10
escala_base = 3
v_base = 3 #df = degrees of freedom, representa o parâmetro (v) da distribuição t




# Função de Máxima Verossimilhança
def Passo_M(parametros):
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
def Passo_E(x):
    u = np.mean(x)
    sigma2 = st.variance(x)
    return (u, sigma2)
    

#Criando vetores para alocar os valores recuperados e posteriormente calcular
#suas médias

locacao_fit = []
escala_fit = []
v_fit = []
amostra_fit = []


#Definindo um contador para verificar quantas iterações serão necessárias até
#a convergência do modelo
contador = 0



while norm > erro:
    for i in range(quant_iter):
        #Amostragem
        data = t.rvs(v_base, size = t_amostra, loc = locacao_base, scale = escala_base)
        
        #Executando o Passo E
        palpite = Passo_E(data)
        
        # "Minimizando" os parâmetros: função, "chute inicial", método utilizado
        # Método usado no cálculo do erro SLSQP = Programação Sequencial de Min Quadrad.
        mv_modelo = minimize(Passo_M, palpite, method='SLSQP', options={'maxiter':n_iter}) 
        
        
        #Guardando os valores ajustados pelo modelo de Máxima Verossimilhança 
        amostra_fit.extend(data)
        locacao_fit.append(mv_modelo.x[0])
        escala_fit.append(mv_modelo.x[1])
    
        # if n == 2:
        #     v_fit.append(v_base)
        # else:
        #     v_fit.append(mv_modelo.x[2])
        

    
    #Calculando o primeiro passo do algoritmo EM
    if contador == 0:
        teta_anterior = np.array([st.mean(locacao_fit), st.mean(escala_fit)])
        # print(teta_anterior)
    else:
        teta_atual = np.array([st.mean(locacao_fit), st.mean(escala_fit)])
        # print(teta_atual)
        teta = teta_atual - teta_anterior
        print(teta)
        norm = np.linalg.norm(teta,2)
        teta_anterior = teta_atual
    contador += 1
        
media_locacao_fit = teta_atual[0]
var_escala_fit = teta_atual[1]


fig, ax = plt.subplots(1, 1)
base = np.linspace(media_locacao_fit - 5*escala_base, media_locacao_fit + 5*escala_base, 1000)
ax.plot(base, t.pdf(base, v_base, loc = media_locacao_fit, scale = var_escala_fit), 'b-', 
        lw=5, alpha=0.6, label='Fit')
ax.plot(base, t.pdf(base, v_base, loc = locacao_base, scale = escala_base), 'r-', 
        lw=5, alpha=0.6, label='Base')

print("\n")
print(rf'A locação da distribuição após o fit foi: u = {media_locacao_fit}')
print(rf'A escala da distribuição após o fit foi: sigma^2 = {var_escala_fit}')
print(rf'O grau de liberdade da distribuição após o fit foi: v = {v_base}')

ax.hist(amostra_fit, bins = 100, density=(True), color = 'skyblue', rwidth=0.9 )
ax.set_xlim(media_locacao_fit - 5*escala_base, media_locacao_fit + 5*escala_base)
ax.legend()
plt.show()
