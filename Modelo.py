import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import mysql.connector
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import psycopg2
import numpy as np
# Conéctate a la base de datos
# Parámetros de conexión
database_params = {
    'host': 'localhost',
    'database': 'DBANALISISPRES',
    'user': 'postgres',
    'password': '123456',
    'port': '5432',  # Puerto predeterminado de PostgreSQL
}
conn = psycopg2.connect(**database_params)

query = "select * from ejecucion_mes"


data = pd.read_sql_query(query, con=conn)


data['fecha']=pd.to_datetime(data['fecha'])
encoder= LabelEncoder()

data['fecha']=encoder.fit_transform(data['fecha'])
X=data['fecha']
y=data.drop('fecha',axis=1)

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.5, random_state=0)
model = LinearRegression()

X_train= np.array(X_train)
X_train= X_train.reshape(-1,1)
model.fit(X_train,y_train)

X_test= np.array(X_test)
X_test= X_test.reshape(-1,1)



y_pred=model.predict(X_test)
data_prueba={
    'fecha':123
}

features= pd.DataFrame(data_prueba, index=[0])

y_pred=model.predict(features)
print(y_pred[0])
quit()
mse = mean_squared_error(y_test, y_pred)

y_test= np.array(y_test)
y_test=y_test.reshape(-1,1)

y_pred= np.array(y_pred)
y_pred=y_pred.reshape(-1,1)

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


quit()

encoder= LabelEncoder()

data['fecha']=encoder.fit_transform(data['fecha'])
empresa = data[(data['empresa'] == 'AAPL') ]

empresa=empresa.drop('empresa',axis=1)
X=empresa.drop('precio',axis=1)


y=empresa['precio']
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=0)
# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()


model.fit(X_train,y_train)

y_pred=model.predict(X_test)
data_prueba={
    'IDStock':562,
    'fecha':123
}
features= pd.DataFrame(data_prueba, index=[0])

y_pred=model.predict(features)
print(y_pred[0])

quit()


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
