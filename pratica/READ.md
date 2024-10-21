# Modelo de Machine Learning: Previsão de Confiabilidade de Clientes para Crédito

Este código implementa um modelo de Machine Learning utilizando o algoritmo Naive Bayes Gaussiano com o objetivo de prever se um cliente de um banco é confiável para receber crédito. O código foi desenvolvido em um ambiente Jupyter Notebook e utiliza as bibliotecas pandas, scikit-learn e yellowbrick.

## Bibliotecas Utilizadas

- **pandas**: Para manipulação de dados.
- **scikit-learn**: Para modelagem de Machine Learning, incluindo ferramentas para pré-processamento, divisão de conjuntos e avaliação do modelo.
- **yellowbrick**: Para visualização da matriz de confusão do modelo.

# Etapas do Código

## 1. Importação e Carregamento de Dados

O arquivo de dados "Credit.csv" é carregado:

```python
credito = pd.read_csv("Credit.csv")
```

## 2. Separação dos Dados

Os dados são separados em duas partes:

- **Previsores**: As características dos clientes (variáveis independentes).
- **Classe**: A variável dependente que indica se o cliente é confiável para crédito (última coluna).

## 3. Codificação de Variáveis Categóricas

O código usa `LabelEncoder` para transformar variáveis categóricas em valores numéricos, essenciais para o algoritmo Naive Bayes:

```python
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
previsores[:, coluna] = labelencoder.fit_transform(previsores[:, coluna])
```

## 4. Divisão do Conjunto de Dados

Os dados são divididos em treinamento (70%) e teste (30%) para avaliar o desempenho do modelo:

```python
from sklearn.model_selection import train_test_split

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores, classe, test_size=0.3, random_state=0)
```

## 5. Treinamento do Modelo

O algoritmo Naive Bayes Gaussiano é utilizado:

```python
from sklearn.naive_bayes import GaussianNB

naive_bayes = GaussianNB()
naive_bayes.fit(X_treinamento, y_treinamento)
```

## 6. Previsão e Avaliação

O modelo faz previsões e a performance é avaliada:

- Previsões

```python
previsoes = naive_bayes.predict(X_teste)
```

- Matriz de Confusão

```python
from sklearn.metrics import confusion_matrix

confusao = confusion_matrix(y_teste, previsoes)
```

- Taxa de Acerto

```python
from sklearn.metrics import accuracy_score

taxa_acerto = accuracy_score(y_teste, previsoes)
```

## 7. Visualização

A matriz de confusão é visualizada com a biblioteca Yellowbrick:

```python
from yellowbrick.classifier import ConfusionMatrix

v = ConfusionMatrix(naive_bayes)
v.fit(X_treinamento, y_treinamento)
v.score(X_teste, y_teste)
v.poof()
```
## 8. Aplicação em Novos Dados

Finalmente, o modelo treinado é utilizado para prever a confiabilidade de novos clientes contidos no arquivo "NovoCredit.csv", após o mesmo pré-processamento.

## Resumo

Este modelo automatiza a análise de crédito de clientes, utilizando técnicas de aprendizado de máquina para prever se um cliente é confiável, com base em características predefinidas.

**Nota**: O uso da matriz de confusão e a visualização fornecem insights sobre o desempenho do modelo e permitem ajustes futuros para aprimoramento.
