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

# Importação do dataset

cellular = read.csv("cellular.csv", sep=";", header=TRUE)
View(cellular)

names(cellular)

dim(cellular)


# Apresentação de head(cellular), definição do fator score_r e apresentação da tabela de frequências absolutas correspondente

# Head do cellular

head(cellular)

# Fatorizar o score_r

cellular$score_r = as.factor(cellular$score_r)

# Tabela de frequências absolutas

table(cellular$score_r)


# Realização de uma análise descritiva dos dados apresentando o número de observações, mínimo, máximo, média, desvio padrão, medida de assimetria e de achatamento

summary(cellular)

describe(cellular)


# Divisão dos dados em amostra de treino (65%) e de teste (35%) usando set.seed(888) e apresentação de tabela de frequências absolutas de score_r em cada amostra

set.seed(888)
ind_treino = sample(nrow(cellular), nrow(cellular)*.65)
cellular_treino = cellular[ind_treino,]
cellular_teste = cellular[-ind_treino,]

# Tabela de frequências absolutas de score_r para data dataset de treino

table(cellular_treino$score_r)

# Tabela de frequências absolutas de score_r para data dataset de teste

table(cellular_teste$score_r)


# Obtenção dos dados dos preditores normalizados (normalização 0-1), nas amostras de treino e teste, e apresentação das primeiras 6 linhas destas amostras após normalização 


# Definição da função de normalização

normalize_min_max <- function(x){
  return ((x - min(x)) / (max(x)-min(x)))}

cellular_treino_n = cellular_treino 
cellular_treino_n[, 1:5] = sapply(cellular_treino_n[,1:5], normalize_min_max)

cellular_teste_n = cellular_teste
cellular_teste_n[, 1:5] = sapply(cellular_teste_n[, 1:5], normalize_min_max)


# Completação das frases seguintes em comentário do script (com eventual obtenção de resultados adicionais)

# 250 
# 7
# 200
# 85.23


# Aprendizagem, sobre a amostra de treino, do 3-Nearest Neighbour (baseado em dois preditores) para prever score_r e avaliação do seu desempenho

# Escolha dos preditores, justificando

# Vamos analisar os preditores mais promissores

correlacoes = round(cor(cellular_treino_n[, 1:6]), 2)
correlacoes[1:5, 6]

# Vamos escolher minutes e bill como os dois melhores preditores pois são os que
# apresentam correlações mais altas com a variável target


# Obtenção do modelo e das correspondentes estimativas de score_r sobre amostra de teste

knn_cellular = knn.reg(cellular_treino_n[, c(1,2)], test = cellular_teste_n[, c(1,2)], y = cellular_treino_n$score, k=3, algorithm="brute")

str(knn_cellular)

pred_knn_cellular = knn_cellular[4]


# Estimativas para o score_r do conjunto de teste

pred_knn_cellular


# Apresentação da Confusion matrix sobre amostra de teste e do índice de Huberty correspondente

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


# Valor mais próximo da primeira observação no conjunto de teste

knn_cellular_cv = knn.cv(cellular[,1:5], cellular$score_r, k=3, prob=TRUE, algorithm="brute")

attr(knn_cellular_cv,"nn.index")[4,]


# #Na aprendizagem foram usados dados normalizados (normalizados/ não normalizados); 
# as observações mais próximas da primeira observação do conjunto de teste são 41, 46, 17 (números das observações); 
# a probabilidade da última observação do conjunto de teste pertencer à classe alvo "No churn", estimada pelo modelo, é  ; 
# segundo os resultados estimados, o churn dos clientes na amostra de teste será 22.73 %.


# Aprendizagem, sobre a amostra de treino, de uma Árvore de Regressão para prever score e avaliação do seu desempenho

# Obtenção do modelo, com cerca de 10 nós folha, e apresentação da árvore correspondente

cellular_tree = tree(cellular_treino$score~. ,data = cellular_treino, control=tree.control(nrow(cellular_treino), mincut = 1, minsize = 2, mindev = 0.001), split = "deviance")
cellular_tree

cellular_tree_10 = prune.tree(cellular_tree, best=10)
cellular_tree_10


# Estimação de score sobre amostra de teste, a partir da árvore obtida, e apresentação das estimativas correspondentes às 6 primeiras observações desta amostra

pred_cellular_tree = predict(cellular_tree_10, cellular_teste)
pred_cellular_tree


# 6 primeiras observações

pred_cellular_tree[1:6]


# Apresentação de 3 métricas de regressão associadas ao modelo aplicado sobre a amostra de teste

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

# Erro de máquina, o erro dado foi argument "data0" is missing, with no default,
# No pc de um membro do grupo deu 0.6277124


# Apresentação, com base nas estimativas obtidas em 3.2), de uma tabela de frequências para as categorias churn e No churn

pred_tree_cellular_r = pred_cellular_tree
pred_tree_cellular_r[pred_tree_cellular_r < 50] = 0
pred_tree_cellular_r[pred_tree_cellular_r >= 50] = 1
pred_tree_cellular_r

tabela_cellular_tree = table(pred_tree_cellular_r)
tabela_cellular_tree

names(tabela_cellular_tree) = c("No churn", "Churn")
tabela_cellular_tree


# Completação das frases seguintes em comentário do script (com eventual obtenção de resultados adicionais

# Na aprendizagem foram usados dados não normalizados (normalizados/ não normalizados); 
# o R-Square associado ao modelo sobre o teste é 0.6277124; 
# o nó folha com menor frequência inclui 2 observações do teste;  
# segundo os resultados estimados, a % de observações da amostra de teste suscetíveis de fazer churn será 14.77.
