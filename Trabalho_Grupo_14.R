# Bibliotecas


library(MASS)
library(Metrics)
library(caret)
library(DescTools)
library(e1071)
library(rcompanion)
library(ISR)
library(lsr)
library(BBmisc)
library(fastDummies)
library(dplyr)
library(mgcv)
library(nnet)
library(tree)
library(psych)
library(FNN)
library(plm)

# Importa��o do dataset

cellular = read.csv("cellular.csv", sep=";", header=TRUE)
View(cellular)

names(cellular)

dim(cellular)


# Apresenta��o de head(cellular), defini��o do fator score_r e apresenta��o da tabela de frequ�ncias absolutas correspondente

# Head do cellular

head(cellular)

# Fatorizar o score_r

cellular$score_r = as.factor(cellular$score_r)

# Tabela de frequ�ncias absolutas

table(cellular$score_r)


# Realiza��o de uma an�lise descritiva dos dados apresentando o n�mero de observa��es, m�nimo, m�ximo, m�dia, desvio padr�o, medida de assimetria e de achatamento

summary(cellular)

describe(cellular)


# Divis�o dos dados em amostra de treino (65%) e de teste (35%) usando set.seed(888) e apresenta��o de tabela de frequ�ncias absolutas de score_r em cada amostra

set.seed(888)
ind_treino = sample(nrow(cellular), nrow(cellular)*.65)
cellular_treino = cellular[ind_treino,]
cellular_teste = cellular[-ind_treino,]

# Tabela de frequ�ncias absolutas de score_r para data dataset de treino

table(cellular_treino$score_r)

# Tabela de frequ�ncias absolutas de score_r para data dataset de teste

table(cellular_teste$score_r)


# Obten��o dos dados dos preditores normalizados (normaliza��o 0-1), nas amostras de treino e teste, e apresenta��o das primeiras 6 linhas destas amostras ap�s normaliza��o 


# Defini��o da fun��o de normaliza��o

normalize_min_max <- function(x){
  return ((x - min(x)) / (max(x)-min(x)))}

cellular_treino_n = cellular_treino 
cellular_treino_n[, 1:5] = sapply(cellular_treino_n[,1:5], normalize_min_max)

cellular_teste_n = cellular_teste
cellular_teste_n[, 1:5] = sapply(cellular_teste_n[, 1:5], normalize_min_max)


# Completa��o das frases seguintes em coment�rio do script (com eventual obten��o de resultados adicionais)

# 250 
# 7
# 200
# 85.23


# Aprendizagem, sobre a amostra de treino, do 3-Nearest Neighbour (baseado em dois preditores) para prever score_r e avalia��o do seu desempenho

# Escolha dos preditores, justificando

# Vamos analisar os preditores mais promissores

correlacoes = round(cor(cellular_treino_n[, 1:6]), 2)
correlacoes[1:5, 6]

# Vamos escolher minutes e bill como os dois melhores preditores pois s�o os que
# apresentam correla��es mais altas com a vari�vel target


# Obten��o do modelo e das correspondentes estimativas de score_r sobre amostra de teste

knn_cellular = knn.reg(cellular_treino_n[, c(1,2)], test = cellular_teste_n[, c(1,2)], y = cellular_treino_n$score, k=3, algorithm="brute")

str(knn_cellular)

pred_knn_cellular = knn_cellular[4]


# Estimativas para o score_r do conjunto de teste

pred_knn_cellular


# Apresenta��o da Confusion matrix sobre amostra de teste e do �ndice de Huberty correspondente

pred_knn_cellular_r = knn_cellular$pred
pred_knn_cellular_r[pred_knn_cellular_r < 50] = 0
pred_knn_cellular_r[pred_knn_cellular_r >= 50] = 1
pred_knn_cellular_r

confusion_mat_ = table(cellular_teste_n$score_r, pred_knn_cellular_r)
confusion_mat_


# Accuracy do modelo

accuracy_cellular = accuracy(cellular_teste_n$score_r, pred_knn_cellular_r)

accuracy_cellular


# Default-p

default_p = max(table(cellular$score_r) / nrow(cellular))
default_p


# Huberty 

cellular_huberty = (accuracy_cellular - default_p) / (1-default_p)
cellular_huberty


# Valor mais pr�ximo da primeira observa��o no conjunto de teste

knn_cellular_cv = knn.cv(cellular[,1:5], cellular$score_r, k=3, prob=TRUE, algorithm="brute")

attr(knn_cellular_cv,"nn.index")[4,]


# #Na aprendizagem foram usados dados normalizados (normalizados/ n�o normalizados); 
# as observa��es mais pr�ximas da primeira observa��o do conjunto de teste s�o 41, 46, 17 (n�meros das observa��es); 
# a probabilidade da �ltima observa��o do conjunto de teste pertencer � classe alvo "No churn", estimada pelo modelo, �  ; 
# segundo os resultados estimados, o churn dos clientes na amostra de teste ser� 22.73 %.


# Aprendizagem, sobre a amostra de treino, de uma �rvore de Regress�o para prever score e avalia��o do seu desempenho

# Obten��o do modelo, com cerca de 10 n�s folha, e apresenta��o da �rvore correspondente

cellular_tree = tree(cellular_treino$score~. ,data = cellular_treino, control=tree.control(nrow(cellular_treino), mincut = 1, minsize = 2, mindev = 0.001), split = "deviance")
cellular_tree

cellular_tree_10 = prune.tree(cellular_tree, best=10)
cellular_tree_10


# Estima��o de score sobre amostra de teste, a partir da �rvore obtida, e apresenta��o das estimativas correspondentes �s 6 primeiras observa��es desta amostra

pred_cellular_tree = predict(cellular_tree_10, cellular_teste)
pred_cellular_tree


# 6 primeiras observa��es

pred_cellular_tree[1:6]


# Apresenta��o de 3 m�tricas de regress�o associadas ao modelo aplicado sobre a amostra de teste

# Deviance

deviance(cellular_tree_10)


# MSE

mse = mse(cellular_teste$score, cellular_tree_10$y)
mse


# MAE

mae = mae(cellular_teste$score, cellular_tree_10$y)
mae


# R-square 

r_square_tree = 1-sum((cellular_teste$score - pred_cellular_tree)^2)/sum((cellular_teste$score - mean(cellular_teste$score))^2)
r_square_tree

# Erro de m�quina, o erro dado foi argument "data0" is missing, with no default,
# No pc de um membro do grupo deu 0.6277124


# Apresenta��o, com base nas estimativas obtidas em 3.2), de uma tabela de frequ�ncias para as categorias churn e No churn

pred_tree_cellular_r = pred_cellular_tree
pred_tree_cellular_r[pred_tree_cellular_r < 50] = 0
pred_tree_cellular_r[pred_tree_cellular_r >= 50] = 1
pred_tree_cellular_r

tabela_cellular_tree = table(pred_tree_cellular_r)
tabela_cellular_tree

names(tabela_cellular_tree) = c("No churn", "Churn")
tabela_cellular_tree


# Completa��o das frases seguintes em coment�rio do script (com eventual obten��o de resultados adicionais

# Na aprendizagem foram usados dados n�o normalizados (normalizados/ n�o normalizados); 
# o R-Square associado ao modelo sobre o teste � 0.6277124; 
# o n� folha com menor frequ�ncia inclui 2 observa��es do teste;  
# segundo os resultados estimados, a % de observa��es da amostra de teste suscet�veis de fazer churn ser� 14.77.
