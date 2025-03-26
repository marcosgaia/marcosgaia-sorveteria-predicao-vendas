import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

# Carregar dados de exemplo (substitua isso com seus dados reais)
# Exemplo: Temperatura (em ºC) e Vendas (quantidade de sorvetes vendidos)
data = {
    "temperatura": [30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
    "vendas": [150, 160, 180, 200, 210, 230, 240, 250, 270, 290]
}

df = pd.DataFrame(data)

# Separar as variáveis independentes (X) e dependentes (y)
X = df[["temperatura"]]  # Temperatura como variável explicativa
y = df["vendas"]        # Vendas como variável alvo

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Calcular o erro quadrático médio (RMSE)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5  # Raiz quadrada do MSE

# Começar a execução do MLflow para registrar o modelo
mlflow.start_run()

# Registrar o modelo com o MLflow
mlflow.sklearn.log_model(model, "modelo_sorvete")

# Registrar a métrica RMSE (erro quadrático médio)
mlflow.log_metric("rmse", rmse)

# Finalizar a execução do MLflow
mlflow.end_run()

print(f"Modelo registrado com RMSE: {rmse}")
