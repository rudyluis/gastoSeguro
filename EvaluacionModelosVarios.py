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
import random
from scipy.stats import norm
resultados=[]

def DatosPredicciones(proyecto):
    ##proyecto='Laguna Colorada'
    query_2 = f"""
        select gestion_presupuestaria,mes,sum(monto_ejecutado) as monto,periodo::integer
        from ejecucion 
        where proyecto='{proyecto}' 
        GROUP BY proyecto,gestion_presupuestaria,mes,periodo
        order by proyecto,gestion_presupuestaria,periodo
        """


    data_query = pd.read_sql_query(query_2, conn)
    print('REGISTROS DE CONSULTA')
    print(data_query['periodo'].count())
    if(data_query['periodo'].count()<=10):
        return 0
    datos_pronostico,datos_comparacion,mse,r2,accuracy,precision,recall,f1,roc_auc,cv_scores,coeficientes,intercepto,x,y=ModeloPrediccion(data_query, proyecto)
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
    # Almacenar resultados en una lista
    resultados.append({
        'Proyecto': proyecto,
        'Datos Pronóstico': datos_pronostico,
        'Datos Comparación': datos_comparacion,
        'MSE': mse,
        'R^2': abs(r2),
        'accuracy':accuracy,
        'precision':precision,
        'recall':recall,
        'f1':f1,
        'roc_auc':roc_auc,
        'cv_scores':cv_scores,
        'coeficientes':coeficientes,
        'intercepto':intercepto,
        'x':x,
        'y':y
    })
    ##plt.show()
   
    ##return fig

def ModeloPrediccion(df,proyecto):
    # Convertir el mes y el año a un formato numérico único para facilitar el modelado
    df['Fecha'] = df['gestion_presupuestaria'].astype(str) + '-' + df['periodo'].astype(str) + '-01'
    # Convertir las fechas a un formato numérico utilizando timestamp
    df['fecha'] = df['Fecha'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').timestamp())
    X = df['fecha'].values.reshape(-1, 1)
    y = df['monto'].values
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Escalar las características
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
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


    x=np.concatenate([y_test, y_pred_test])
    y=np.concatenate([['Valores Reales'] * len(y_test), ['Predicciones'] * len(y_pred_test)])
 


    return df_predicciones, df_predicciones_comparacion,mse,r2,accuracy,precision,recall,f1,roc_auc,cv_scores,coeficientes,intercepto,x, y


if __name__ == "__main__":

    database_params = {
        'host': 'localhost',
        'database': 'DBANALISISPRES',
        'user': 'postgres',
        'password': '123456',
        'port': '5432',  # Puerto predeterminado de PostgreSQL
    }
    conn = psycopg2.connect(**database_params)
    query_proyecto="Select distinct proyecto from ejecucion order by proyecto"

    proyectos  = pd.read_sql_query(query_proyecto, conn)


    proyectos_list = proyectos['proyecto'].tolist()

    # Generar un número entero aleatorio entre 1 y 10 (inclusive)

    # Crear una única figura para todos los gráficos
    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(15, 20))  # 5 filas, 2 columnas para 10 proyectos
    fig2, axs2 = plt.subplots(nrows=5, ncols=2, figsize=(15, 20))  # 5 filas, 2 columnas para 10 proyectos
    fig3, axs3 = plt.subplots(nrows=5, ncols=2, figsize=(15, 20))
    # Inicializar el índice de la subfigura
    subfig_idx = 0


    for n_des in range(1, 16):
        n_proyec = random.randint(0, len(proyectos_list) - 1)
        proyecto_seleccionado = proyectos_list[n_proyec]
        print(f"Proyecto seleccionado ({n_des}): {proyecto_seleccionado}")
        DatosPredicciones(proyecto_seleccionado)
        
    print("RESULTADOS")
    print(resultados)
    # Guardar resultados en un archivo de texto


    with open('resultados.txt', 'w',encoding='utf-8') as archivo:
        for resultado in resultados:
            proyecto=resultado['Proyecto']
            archivo.write(f"Resultados para el Proyecto {resultado['Proyecto']}:\n")
            archivo.write(f"MSE: {resultado['MSE']}\n")
            archivo.write(f"R^2: {resultado['R^2']}\n")
            archivo.write(f"Accuracy: {resultado['accuracy']}\n")
            archivo.write(f"Precision: {resultado['precision']}\n")
            archivo.write(f"Recall: {resultado['recall']}\n")
            archivo.write(f"F1: {resultado['f1']}\n")
            archivo.write(f"roc_auc: {resultado['roc_auc']}\n")
            archivo.write(f"cv_scores: {resultado['cv_scores']}\n")
            archivo.write(f"Coeficientes: {resultado['coeficientes']}\n")
            archivo.write(f"Intercepto: {resultado['intercepto']}\n")
            archivo.write(f"=======================================\n")

            sns.boxplot(ax=axs[subfig_idx // 2, subfig_idx % 2], x=resultado['x'],y=resultado['y'])
            axs[subfig_idx // 2, subfig_idx % 2].set_title(f'Proy.: {proyecto}',fontsize=7)
            
            ##grafica de Lineas
            axs2[subfig_idx // 2, subfig_idx % 2].plot(resultado['x'], label='Valores Reales', marker='o')
            axs2[subfig_idx // 2, subfig_idx % 2].plot(resultado['y'], label='Predicciones', marker='o')
            axs2[subfig_idx // 2, subfig_idx % 2].set_title('Valores Reales y Predicciones')
            axs2[subfig_idx // 2, subfig_idx % 2].set_xlabel('Índice')
            axs2[subfig_idx // 2, subfig_idx % 2].set_ylabel('Valor')
            axs2[subfig_idx // 2, subfig_idx % 2].legend()

            # Curva Gaussiana
            x_range = np.linspace(min(resultado['x']), max(resultado['x']), 100)
            mu, std = norm.fit(resultado['x'])
            gaussiana = norm.pdf(x_range, mu, std)
            axs3[subfig_idx // 2, subfig_idx % 2].plot(x_range, gaussiana, color='green', label='Curva Gaussiana Proyecto:'+str(subfig_idx+1))
            axs3[subfig_idx // 2, subfig_idx % 2].legend()

            subfig_idx+=1


    for ax in axs.flat:
        ax.set(xlabel='x-label', ylabel='y-label')
        ax.tick_params(axis='both', labelsize=8) 
    plt.tight_layout()  # Ajustar el diseño de la figura
    plt.show()  # Mostrar la figura con todos los gráficos
    print("Resultados guardados en 'resultados.txt'")


    # Calcular el promedio de la MSE
    mse_promedio= np.mean([resultado['MSE']for resultado in resultados])
    r2_promedio = np.mean([resultado['R^2'] for resultado in resultados])
    accuracy = np.mean([resultado['accuracy'] for resultado in resultados])
    precision = np.mean([resultado['precision'] for resultado in resultados])
    recall = np.mean([resultado['recall'] for resultado in resultados])
          
    # Imprimir el resultado
    print(f"MSE : {mse_promedio}")
    print(f"R2 para todos los proyectos: {r2_promedio}")
    print(f"Accuracy para todos los proyectos: {accuracy}")
    print(f"Precision para todos los proyectos: {precision}")
    print(f"Recall para todos los proyectos: {recall}")
    quit()


