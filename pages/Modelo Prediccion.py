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

def ModeloSARIMA(df):
    # Convertir las fechas a formato datetime
    df['Fecha'] = df['gestion_presupuestaria'].astype(str) + '-' + df['periodo'].astype(str) + '-01'
    
    # Convertir las fechas a un formato numérico utilizando timestamp
    df['fecha'] = df['Fecha'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').timestamp())
   

    # Dividir los datos en características (X) y etiquetas (y)
    y = df['monto']
    print(y)
    # Configurar el modelo SARIMA
    modelo = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

    # Ajustar el modelo a los datos
    resultado = modelo.fit(disp=False)

    # Generar un rango de fechas desde la última fecha en el DataFrame hasta diciembre de 2023
    ultima_fecha = df['fecha'].max()
    fechas_futuras = pd.date_range(start=ultima_fecha, end='2023-12-01', freq='MS')  # MS: Month Start

    # Realizar predicciones para todas las fechas futuras
    predicciones_futuras = resultado.get_forecast(steps=len(fechas_futuras))
    predicciones_futuras_ci = predicciones_futuras.conf_int()

    # Crear un DataFrame con las fechas y las predicciones
    df_predicciones = pd.DataFrame({'Fecha': fechas_futuras, 'Prediccion': predicciones_futuras.predicted_mean.values,
                                    'Lower_CI': predicciones_futuras_ci.iloc[:, 0].values,
                                    'Upper_CI': predicciones_futuras_ci.iloc[:, 1].values})

    # Imprimir el DataFrame de predicciones
    print(df_predicciones)
   
    # Valores para comparar con los actuales
    primera_fecha = df['fecha'].min()
    ultima_fecha = df['fecha'].max()
    fechas_comparacion = pd.date_range(start=primera_fecha, end=ultima_fecha, freq='MS')  # MS: Month Start

    # Realizar predicciones para todas las fechas de comparación
    predicciones_comparacion = resultado.get_prediction(start=0, end=len(y) - 1)
    predicciones_comparacion_ci = predicciones_comparacion.conf_int()

    # Crear un DataFrame con las fechas de comparación y las predicciones
   
    # Imprimir el DataFrame de comparación
    
    return df_predicciones, df_predicciones


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
    print(df_predicciones)
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
    print(df_predicciones_comparacion)
    return df_predicciones, df_predicciones_comparacion


def ModeloPrediccionGeneral(df):
      # Convertir las fechas a formato datetime
    df['fecha'] = pd.to_datetime(df['fecha'])

    # Dividir los datos en características (X) y etiquetas (y)
    X = df['fecha'].values.reshape(-1, 1)
    y = df.drop('fecha', axis=1)

    # Escalar las características
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

    # Crear un modelo de regresión lineal
    modelo = LinearRegression()

    # Entrenar el modelo
    modelo.fit(X_train, y_train)

    # Generar un rango de fechas desde la última fecha en el DataFrame hasta diciembre de 2023
    ultima_fecha = df['fecha'].max()
    fechas_futuras = pd.date_range(start=ultima_fecha, end='2023-12-01', freq='MS')  # MS: Month Start

    # Asegurarse de que las fechas futuras estén en el mismo formato que las fechas originales
    fechas_futuras_numericas = fechas_futuras.map(lambda x: datetime.timestamp(x))

    # Escalar las fechas futuras
    fechas_futuras_scaled = scaler.transform(fechas_futuras_numericas.values.reshape(-1, 1))

    # Realizar predicciones para todas las fechas futuras
    predicciones_futuras = modelo.predict(fechas_futuras_scaled)

    # Crear un DataFrame con las fechas y las predicciones
    df_predicciones = pd.DataFrame({'Fecha': fechas_futuras, 'Prediccion': predicciones_futuras[:, 0]})

    # Imprimir el DataFrame de predicciones
    print(df_predicciones)

    # Valores para comparar con los actuales
    primera_fecha = df['fecha'].min()
    ultima_fecha = df['fecha'].max()
    fechas_comparacion = pd.date_range(start=primera_fecha, end=ultima_fecha, freq='MS')  # MS: Month Start
    fechas_comparacion_numericas = fechas_comparacion.map(lambda x: datetime.timestamp(x))

    # Escalar las fechas de comparación
    fechas_comparacion_scaled = scaler.transform(fechas_comparacion_numericas.values.reshape(-1, 1))

    # Realizar predicciones para todas las fechas de comparación
    predicciones_comparacion = modelo.predict(fechas_comparacion_scaled)

    # Crear un DataFrame con las fechas de comparación y las predicciones
    predicciones_comparacion_abs = np.abs(predicciones_comparacion)
    df_predicciones_comparacion = pd.DataFrame({'Fecha': fechas_comparacion, 'Prediccion': predicciones_comparacion_abs[:, 0]})

    # Imprimir el DataFrame de comparación
    print(df_predicciones_comparacion)

    return df_predicciones, df_predicciones_comparacion




encoder= LabelEncoder()
st.set_page_config(page_title="Gasto Seguro - Modelo Prediccion",
                   page_icon="📈",
                   layout="centered",
                   initial_sidebar_state="auto"
                   )
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
st.title("💵 Modelo de Predicción")
st.divider()

##logo= "imagen.jpg"
##st.sidebar.image(logo, width=100)



database_params = {
    'host': 'localhost',
    'database': 'DBANALISISPRES',
    'user': 'postgres',
    'password': '123456',
    'port': '5432',  # Puerto predeterminado de PostgreSQL
}
conn = psycopg2.connect(**database_params)

st.sidebar.header("🔼Seleccione Opciones")


query = "select * from ejecucion order by proyecto,gestion_presupuestaria"
data = pd.read_sql_query(query, conn)
datos=data.copy()
datos_proyecto= ['Todos los Proyectos'] + list(datos["proyecto"].unique())
proyecto = st.sidebar.selectbox("Seleccione el Proyecto", datos_proyecto,help="Elige una proyecto.",placeholder="Elije un Proyecto")
datos_gestion= datos[datos['proyecto']==proyecto]
gestion= st.sidebar.selectbox('Gestion',datos_gestion['gestion_presupuestaria'].unique())
if(not(proyecto=='Todos los Proyectos')):
    query_2 = f"""
    select * from ejecucion 
    where proyecto='{proyecto}' and gestion_presupuestaria='{gestion}'
    order by proyecto,gestion_presupuestaria
    """

    query_2 = f"""
    select gestion_presupuestaria,mes,sum(monto_ejecutado) as monto,periodo::integer
    from ejecucion 
    where proyecto='{proyecto}' 
    GROUP BY proyecto,gestion_presupuestaria,mes,periodo
    order by proyecto,gestion_presupuestaria,periodo
    """


    data_query = pd.read_sql_query(query_2, conn)
    print(data_query)

    st.subheader(f" 📋Datos Originales")
    AgGrid(data_query,height=200, width=500)
    
    data_query['Fecha'] = data_query['gestion_presupuestaria'].astype(str) + '-' + data_query['periodo'].astype(str) + '-01'
    

    chart1=alt.Chart(data_query).mark_line(color='red').encode(
            x=alt.X('Fecha:T',axis=alt.Axis(title='Fecha')),
            y=alt.Y('monto:Q',axis=alt.Axis(title='Monto')),
            tooltip=[alt.Tooltip('Fecha:T', title='Fecha'), alt.Tooltip('monto:Q', title='Monto')]

    ).properties(
            width=600,
            height=300
        )
    
    st.altair_chart(chart1,use_container_width=True)
else:    
    query_2 = f"""
    select * from ejecucion_mes
    """
    data_query = pd.read_sql_query(query_2, conn)
    print(data_query)

    st.subheader(f" 📋Datos Originales")
    AgGrid(data_query,height=200, width=500)
    

    chart1=alt.Chart(data_query).mark_line(color='red').encode(
            x=alt.X('fecha:T',axis=alt.Axis(title='Fecha')),
            y=alt.Y('recurso:Q',axis=alt.Axis(title='Monto Recurso')),
            tooltip=[alt.Tooltip('Fecha:T', title='Fecha'), alt.Tooltip('recurso:Q', title='Monto Recurso')]

    ).properties(
            width=600,
            height=300
    )
    chart2=alt.Chart(data_query).mark_line(color='blue').encode(
            x=alt.X('fecha:T',axis=alt.Axis(title='Fecha')),
            y=alt.Y('inversion:Q',axis=alt.Axis(title='Monto Recurso')),
            tooltip=[alt.Tooltip('Fecha:T', title='Fecha'), alt.Tooltip('inversion:Q', title='Monto Recurso')]

    ).properties(
            width=600,
            height=300
    )
    chart3=alt.Chart(data_query).mark_line(color='green').encode(
            x=alt.X('fecha:T',axis=alt.Axis(title='Fecha')),
            y=alt.Y('gasto:Q',axis=alt.Axis(title='Monto Recurso')),
            tooltip=[alt.Tooltip('Fecha:T', title='Fecha'), alt.Tooltip('gasto:Q', title='Monto Recurso')]

    ).properties(
            width=600,
            height=300
    )
    st.altair_chart(chart1+chart2+chart3,use_container_width=True)


    
st.sidebar.header("🔼Pronostico")
opcion_pred= st.sidebar.selectbox('Opcion',['Gestion(Actual)','Mensual(Todos Los Meses)'])
Regresion= st.sidebar.checkbox("Regresion Lineal", value=True)
Normalizacion= st.sidebar.checkbox("Normalizacion", value=True)


# Crear un botón con una etiqueta y asignarle la acción
if st.sidebar.button("Aplicar Prediccion"):

    if(not(proyecto=='Todos los Proyectos')):
        ### para un proyecto
        datos_pronostico,datos_comparacion=ModeloPrediccion(data_query)
        st.subheader(f" 📋Datos Pronosticados")
        AgGrid(datos_pronostico,height=200, width=500)
        

        chart2=alt.Chart(datos_pronostico).mark_line().encode(
            x=alt.X('Fecha:T',axis=alt.Axis(title='Fecha')),
            y=alt.Y('Prediccion:Q',axis=alt.Axis(title='Monto')),
            tooltip=[alt.Tooltip('Fecha:T', title='Fecha'), alt.Tooltip('Prediccion:Q', title='Monto')]

        ).properties(
            width=600,
            height=300
        )
    
        st.altair_chart(chart2,use_container_width=True)

        st.subheader(f"📈Datos Comparacion")
        AgGrid(datos_comparacion,height=200, width=500)
        

        chart3=alt.Chart(datos_comparacion).mark_line().encode(
            x=alt.X('Fecha:T',axis=alt.Axis(title='Fecha')),
            y=alt.Y('Prediccion:Q',axis=alt.Axis(title='Monto')),
            tooltip=[alt.Tooltip('Fecha:T', title='Fecha'), alt.Tooltip('Prediccion:Q', title='Monto')]

        ).properties(
            width=600,
            height=300
        )
    
        st.altair_chart(chart3,use_container_width=True)
        combined_chart = chart1 + chart3
        st.altair_chart(combined_chart,use_container_width=True)
    else:
        #### cuando es todos
        datos_pronostico,datos_comparacion=ModeloPrediccionGeneral(data_query)
        st.subheader(f" 📋Datos Pronosticados para todos los Proyectos")
        AgGrid(datos_pronostico,height=200, width=500)
        

        chart2=alt.Chart(datos_pronostico).mark_line().encode(
            x=alt.X('Fecha:T',axis=alt.Axis(title='Fecha')),
            y=alt.Y('Prediccion:Q',axis=alt.Axis(title='Monto')),
            tooltip=[alt.Tooltip('Fecha:T', title='Fecha'), alt.Tooltip('Prediccion:Q', title='Monto')]

        ).properties(
            width=600,
            height=300
        )
    
        st.altair_chart(chart2,use_container_width=True)

        st.subheader(f"📈Datos Comparacion")
        AgGrid(datos_comparacion,height=200, width=500)
        

        chart3=alt.Chart(datos_comparacion).mark_line().encode(
            x=alt.X('Fecha:T',axis=alt.Axis(title='Fecha')),
            y=alt.Y('Prediccion:Q',axis=alt.Axis(title='Monto')),
            tooltip=[alt.Tooltip('Fecha:T', title='Fecha'), alt.Tooltip('Prediccion:Q', title='Monto')]

        ).properties(
            width=600,
            height=300
        )
    
        st.altair_chart(chart3,use_container_width=True)
        combined_chart = chart1 + chart3
        st.altair_chart(combined_chart,use_container_width=True)
quit()
