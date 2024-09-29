# Importar bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar el conjunto de datos de diabetes
diabetes = datasets.load_diabetes()
print(diabetes)
X = diabetes.data[:, np.newaxis, 2]  # Utilizamos solo una característica para simplificar (columna 2)
y = diabetes.target

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular el error cuadrático medio
mse = mean_squared_error(y_test, y_pred)

# Imprimir los coeficientes y el error
print("Coeficiente:", model.coef_[0])
print("Intercepto:", model.intercept_)
print("Error Cuadrático Medio:", mse)

# Visualizar los resultados
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.title('Regresión Lineal Simple')
plt.xlabel('Característica')
plt.ylabel('Target')
plt.show()

