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
import math


'''
O objetivo deste passo é encontrar a máxima verossimilhança de amostras t-student
de tamanho "t_amostra", repetindo o processo "quant_iter" vezes. Para isso, 
utilizaremos a função minimize da biblioteca scipy.optimize e iremos tentar 
reduzir o valor do oposto da soma das logpdf. Com isso, conseguimos recuperar
os valores de mu, sigma^2 e v da t-student original (de acordo com o tamanho
da amostra).
'''

#Definindo o erro para a parada do algoritmo EM
erro = 1e-5

#Criando a variável que receberá a norma dos parâmetros
norm = 1 

#Definir o tamanho das amostras que serão avaliadas
t_amostra = 50

#Variável para o critério de parada
parada = 0

#Definindo o número máximo de iterações
max_iter = 10000

#Inserir n = 2 se deseja estimar mu e sigma2. Caso queira estimar o parâmetro v
#defina n com qualquer valor diferente de 2
n = 2

#Definindo o número máximo de iterações a ser considerada na função "minimize"
n_iter = 200


# Gerando uma amostra t-student com locação e escala (diferentes de 0 e 1) 
locacao_base = 32
escala_base = 5
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


#Gerando o primeiro palpite do Passo E
def Passo_E0(x):
    u = np.mean(x)
    sigma2 = st.variance(x)
    return (u, sigma2)


#Calculando os palpites posteriores
def Passo_E(y):
    u = st.mean(y[0])
    sigma2 = st.mean(y[1])
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



while parada == 0 :

    #Amostragem
    data = t.rvs(v_base, size = t_amostra, loc = locacao_base, scale = escala_base)
    
    
    #Executando o Passo E
    if contador == 0:
        
        palpite_inicial = Passo_E0(data) #Primeiro Passo E (palpite dado a partir dos dados)
        mv_modelo = minimize(Passo_M, palpite_inicial, method='SLSQP', options={'maxiter':n_iter})
        
        #Guardando os valores ajustados pelo modelo de Máxima Verossimilhança 
        amostra_fit.extend(data)
        locacao_fit.append(mv_modelo.x[0])
        escala_fit.append(mv_modelo.x[1])
        
        theta_anterior = np.array([st.mean(locacao_fit), st.mean(escala_fit)])
    else:
        y = np.array([locacao_fit, escala_fit])
        palpite = Passo_E(y) #Primeiro Passo E (palpite dado a partir dos dados)
        # "Minimizando" os parâmetros: função, "chute inicial", método utilizado
        # Método usado no cálculo do erro SLSQP = Programação Sequencial de Min Quadrad.
        mv_modelo = minimize(Passo_M, palpite, method='SLSQP', options={'maxiter':n_iter})
        
        
        #Guardando os valores ajustados pelo modelo de Máxima Verossimilhança 
        amostra_fit.extend(data)
        locacao_fit.append(mv_modelo.x[0])
        escala_fit.append(mv_modelo.x[1])
        
        y = np.array([locacao_fit, escala_fit])
        
        theta_atual = np.array(Passo_E(y))
        
        log_theta_atual = np.log10(theta_atual)
        log_theta_anterior = np.log10(theta_anterior)
        
        teta = log_theta_atual - log_theta_anterior

        norm = np.linalg.norm(teta,2)
        
        theta_anterior = theta_atual
        
        if(norm < erro) or (contador > max_iter):
            parada = 1
        
        # print(norm)
    

    contador += 1
    print(contador)
        
media_locacao_fit = theta_atual[0]
var_escala_fit = theta_atual[1]

print('\n\n')
print(f'Após {contador} a sequência convergiu.')


fig, ax = plt.subplots(1, 1)
base = np.linspace(media_locacao_fit - 5*escala_base, media_locacao_fit + 5*escala_base, 1000)
ax.plot(base, t.pdf(base, v_base, loc = media_locacao_fit, scale = var_escala_fit), 'b-', 
        lw=8, alpha=0.6, label='Fit')
ax.plot(base, t.pdf(base, v_base, loc = locacao_base, scale = escala_base), 'r-', 
        lw=3, alpha=0.6, label='Base')

print("\n\n")
print(rf'A locação da distribuição após o fit foi: u = {media_locacao_fit}')
print(rf'A escala da distribuição após o fit foi: sigma^2 = {var_escala_fit}')
print(rf'O grau de liberdade da distribuição após o fit foi: v = {v_base}')

ax.hist(amostra_fit, bins = 500, density=(True), color = 'skyblue', rwidth=0.9, label = f'Amostras (n = {t_amostra})' )
print(len(amostra_fit))
ax.set_xlim(media_locacao_fit - 5*escala_base, media_locacao_fit + 5*escala_base)
ax.set_title('Histograma do Fit')
ax.legend()
plt.show()


plt.figure('Medias das amostras')
plt.title(f'Médias amostrais (n = {t_amostra})')
plt.hist(locacao_fit, bins = 50, density=(True), label=r'$\mu$')
plt.legend()



plt.figure('Variancia das amostras')
plt.title(f'Variâncias amostrais (n = {t_amostra})')
plt.hist(escala_fit, bins = 50, density=(True), label=r'$\sigma^2$')
plt.legend()