# marcosgaia-sorveteria-predicao-vendas


# Desafio: Prevendo Vendas de Sorvete com Machine Learning 🍦
Este projeto visa prever as vendas de sorvete de uma sorveteria, a Gelato Mágico, com base na temperatura ambiente. Através de um modelo de Machine Learning, é possível otimizar a produção de sorvetes e reduzir desperdícios. O desafio envolve a implementação de um modelo de regressão, o uso do MLflow para monitoramento e o uso de pipeline para garantir reprodutibilidade.

# Objetivo
Desenvolver um modelo de regressão preditiva para prever as vendas de sorvetes com base na temperatura do dia e registrar o modelo com MLflow.

# Passos para Rodar o Projeto no WSL
Instalar o WSL no Windows (caso ainda não tenha feito):

O WSL permite rodar um ambiente Linux no Windows. Para configurá-lo, siga o tutorial oficial da Microsoft: Instalar o WSL.

Após a instalação, você pode abrir o WSL no seu terminal, onde você terá um terminal Linux funcional.

Instalar o Python no WSL: No terminal do WSL, certifique-se de ter o Python instalado:

bash
Copiar
sudo apt update
sudo apt install python3 python3-pip
Criar e Ativar um Ambiente Virtual: Para garantir que todas as dependências do projeto estejam isoladas, criamos um ambiente virtual:

bash
Copiar
python3 -m venv venv
source venv/bin/activate
Instalar as Dependências: Em seguida, instale todas as dependências necessárias para o projeto. Certifique-se de instalar o scikit-learn, MLflow, e outras bibliotecas que você usará.

bash
Copiar
pip install scikit-learn mlflow pandas matplotlib numpy
Configuração do Projeto:

Dentro do WSL, crie a estrutura de pastas para o projeto e adicione arquivos como modelo.py, o arquivo README.md, etc.

Crie a pasta inputs para armazenar qualquer dado de entrada ou exemplos necessários.

Criação do Modelo de Machine Learning: O modelo de Machine Learning foi implementado em um script Python, modelo.py, que utiliza o scikit-learn para treinar o modelo de regressão. O código principal para treinar o modelo e calcular a RMSE (Root Mean Squared Error) está abaixo:

python
Copiar
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

# Carregar os dados (substitua isso por seu próprio conjunto de dados)
data = pd.read_csv('dados_sorvete.csv')  # Exemplo de dado

# Prepara os dados
X = data[['temperatura']]  # Exemplo de feature
y = data['vendas']  # Exemplo de target

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Calcular o RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Registrar o modelo e as métricas no MLflow
mlflow.start_run()
mlflow.sklearn.log_model(model, "modelo_sorvete")
mlflow.log_metric("rmse", rmse)
mlflow.end_run()
Registrar o Modelo no MLflow: Para acompanhar e versionar o modelo, utilizamos o MLflow, que foi integrado ao código no passo anterior. Isso permite registrar o modelo e as métricas associadas, como o RMSE, que são logadas no MLflow.

Criação do Repositório no GitHub: Você criou um repositório no GitHub para armazenar o código e registrar o histórico do projeto. Use os comandos Git para versionar o código:

bash
Copiar
git init
git add .
git commit -m "Adiciona modelo de previsão de vendas de sorvete"
git remote add origin <URL_DO_SEU_REPOSITORIO_GITHUB>
git push -u origin main
Compartilhamento do Link do Repositório: Após finalizar a implementação e garantir que o código esteja funcionando corretamente, você compartilhou o link do repositório no ambiente de entrega (provavelmente na DIO ou outra plataforma de cursos) para avaliação.

Arquivos Importantes do Projeto:
modelo.py: Contém a implementação do modelo de Machine Learning com a regressão linear e o registro no MLflow.

README.md: Este arquivo que contém detalhes sobre o desafio, como configurar o ambiente e executar o projeto.

inputs/: Pasta contendo quaisquer dados ou exemplos usados para treinar o modelo.

dados_sorvete.csv: Exemplo de dataset utilizado para treinar o modelo. Este arquivo pode variar dependendo dos dados reais que você usou.

Problemas Encontrados e Soluções:
Durante o processo de implementação no WSL, alguns problemas foram encontrados, incluindo:

Erro ao tentar instalar pacotes: Caso tenha encontrado dificuldades na instalação de pacotes devido a problemas de permissões, uma solução pode ser criar um ambiente virtual e ativá-lo antes de instalar pacotes.

Problemas com arquivos grandes no Git: Algumas bibliotecas ou pacotes podem ser grandes demais para o GitHub. Use o Git LFS (Large File Storage) se for necessário, ou então ignore arquivos grandes, como arquivos binários.

Erro ao tentar rodar o modelo: Certifique-se de que todos os pacotes necessários estão instalados corretamente, e que o código está corretamente configurado para ler os dados de entrada.

Este README.md serve como guia para explicar o meu processo de configuração e implementação do projeto. Com ele, outras pessoas poderão entender como configurar o ambiente e executar o código com sucesso, além de como tive que lidar com as dificuldades usando o WSL para rodar o projeto.
