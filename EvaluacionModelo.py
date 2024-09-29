import streamlit as st
import pandas as pd
import numpy as np
import pickle
import altair as alt
from sklearn.preprocessing import LabelEncoder
from pandas import set_option
from st_aggrid import AgGrid
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import psycopg2
from dateutil import parser
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import binarize
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
def ModeloPrediccion(df):
    # Convertir el mes y el año a un formato numérico único para facilitar el modelado
    df['Fecha'] = df['gestion_presupuestaria'].astype(str) + '-' + df['periodo'].astype(str) + '-01'
    
    # Convertir las fechas a un formato numérico utilizando timestamp
    df['fecha'] = df['Fecha'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').timestamp())
    
    X = df['fecha'].values.reshape(-1, 1)
    y = df['monto'].values

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear un modelo de regresión lineal
    modelo = LinearRegression()

    # Entrenar el modelo
    modelo.fit(X_train, y_train)

    # Generar un rango de fechas desde la última fecha en el DataFrame hasta diciembre de 2023
    ultima_fecha = df['Fecha'].max()
    fechas_futuras = pd.date_range(start=ultima_fecha, end='2023-12-01', freq='MS')  # MS: Month Start
    fechas_futuras = pd.date_range(start=fechas_futuras[1], end='2023-12-01', freq='MS')  # MS: Month Start

    # Asegurarse de que las fechas futuras estén en el mismo formato que las fechas originales
    fechas_futuras_numericas = fechas_futuras.map(lambda x: datetime.timestamp(x))

    # Realizar predicciones para todas las fechas futuras
    predicciones_futuras = modelo.predict(fechas_futuras_numericas.values.reshape(-1, 1))

    # Crear un DataFrame con las fechas y las predicciones
    df_predicciones = pd.DataFrame({'Fecha': fechas_futuras, 'Prediccion': predicciones_futuras})

    # Imprimir el DataFrame de predicciones
    ## valores para compara con los actuales
    primera_fecha = df['Fecha'].min()
    ultima_fecha = df['Fecha'].max()
    fechas_futuras = pd.date_range(start=primera_fecha, end=ultima_fecha, freq='MS')  # MS: Month Start
   
    # Asegurarse de que las fechas futuras estén en el mismo formato que las fechas originales
    fechas_futuras_numericas = fechas_futuras.map(lambda x: datetime.timestamp(x))

    # Realizar predicciones para todas las fechas futuras
    predicciones_futuras = modelo.predict(fechas_futuras_numericas.values.reshape(-1, 1))

    # Crear un DataFrame con las fechas y las predicciones
    df_predicciones_comparacion= pd.DataFrame({'Fecha': fechas_futuras, 'Prediccion': predicciones_futuras})
    ##print(df_predicciones_comparacion)

    y_pred_test = modelo.predict(X_test)

    # Calcular métricas de regresión
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    
    # Convertir a una tarea de clasificación binaria (umbral de 0.5)
    y_pred_class = binarize([y_pred_test], threshold=0.5)[0]

    # Convertir y_test a una tarea de clasificación binaria (umbral de 0.5)
    y_test_class = binarize([y_test], threshold=0.5)[0]

    # Calcular métricas de clasificación
    accuracy = accuracy_score(y_test_class, y_pred_class)
    precision = precision_score(y_test_class, y_pred_class)
    recall = recall_score(y_test_class, y_pred_class)
    f1 = f1_score(y_test_class, y_pred_class)

    # Calcular la curva ROC y el área bajo la curva (AUC-ROC)
    fpr, tpr, _ = roc_curve(y_test_class, y_pred_test)
    roc_auc = auc(fpr, tpr)

     # Realizar validación cruzada
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=5)  # 5-fold cross-validation
   
   
    coeficientes = modelo.coef_
    intercepto = modelo.intercept_
    
   

    plt.figure(figsize=(10, 6))
    sns.boxplot(x=np.concatenate([y_test, y_pred_test]), y=np.concatenate([['Valores Reales'] * len(y_test), ['Predicciones'] * len(y_pred_test)]))

    plt.title('Comparación de Valores Reales y Predicciones')
    plt.show()

    return df_predicciones, df_predicciones_comparacion,mse,r2,accuracy,precision,recall,f1,roc_auc,cv_scores,coeficientes,intercepto

if __name__ == "__main__":

    database_params = {
        'host': 'localhost',
        'database': 'DBANALISISPRES',
        'user': 'postgres',
        'password': '123456',
        'port': '5432',  # Puerto predeterminado de PostgreSQL
    }
    conn = psycopg2.connect(**database_params)
    proyecto='Laguna Colorada'
    query_2 = f"""
        select gestion_presupuestaria,mes,sum(monto_ejecutado) as monto,periodo::integer
        from ejecucion 
        where proyecto='{proyecto}' 
        GROUP BY proyecto,gestion_presupuestaria,mes,periodo
        order by proyecto,gestion_presupuestaria,periodo
        """


    data_query = pd.read_sql_query(query_2, conn)
    print(data_query)
    datos_pronostico,datos_comparacion,mse,r2,accuracy,precision,recall,f1,roc_auc,cv_scores,coeficientes,intercepto=ModeloPrediccion(data_query)
    print("DATOS PRONOSTICO")
    print(datos_pronostico)

    print("DATOS COMPARACION")
    print(datos_comparacion)

    print("Métricas de Regresión:")
    print(f"Error Cuadrático Medio (MSE): {mse}")
    print(f"Coeficiente de Determinación (R^2): {r2}")
        

    print("Métricas de Clasificación:")
    print(f"Exactitud (Accuracy): {accuracy}")
    print(f"Precisión: {precision}")
    print(f"Exhaustividad (Recall): {recall}")
    print(f"Puntuación F1: {f1}")
    print(f"AUC-ROC: {roc_auc}")    

    # Imprimir métricas de validación cruzada
    print("Métricas de Validación Cruzada:")
    print(f"Puntuaciones de validación cruzada: {cv_scores}")
    print(f"Media de puntuaciones de validación cruzada: {np.mean(cv_scores)}")


    print("Coeficientes del modelo:")
    for i, coef in enumerate(coeficientes):
        print(f"Coeficiente {i+1}: {coef}")

    print(f"Intercepto: {intercepto}")

