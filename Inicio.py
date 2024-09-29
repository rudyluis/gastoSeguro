import streamlit as st
from st_aggrid import AgGrid
##import plotly.express as px
import altair as alt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components
#import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
##from sqlalchemy import create_engine
import psycopg2
from pivottablejs import pivot_ui
# Parámetros de conexión
database_params = {
      'host': 'dpg-crsbjetds78s73e3d0f0-a.ohio-postgres.render.com',
    'database': 'gastoseguro',
    'user': 'rmanzanedav',
    'password': 'ZG7LFQBeEdQQxxZONAJCgV8Fcm2Y8ts1',
    'port': '5432',  # Puerto predeterminado de PostgreSQL
}
##conn = psycopg2.connect(**database_params)
# Crear una conexión utilizando SQLAlchemy
##engine = create_engine(f'postgresql://{database_params["user"]}:{database_params["password"]}@{database_params["host"]}:{database_params["port"]}/{database_params["database"]}')

# Ejecutar una consulta y cargar los resultados en un DataFrame de Pandas
query = "select * from ejecucion where gestion_presupuestaria::integer>=2019"

##data = pd.read_sql_query(query, conn)
##data.to_csv('df/ejecucion.csv', index=False)

data = pd.read_csv('df/ejecucion.csv',delimiter=',',on_bad_lines='skip', encoding='ISO-8859-1')
st.set_page_config(page_title="CMI GASTO SEGURO", page_icon="📈",layout="wide")

st.title("📈 GASTO SEGURO")
st.header("MODELO DE PREDICCION PARA EMPRESAS PUBLICAS")
st.subheader(" :bar_chart: CMI Gestion Financiera")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
st.divider()
st.subheader("""💵Tabla Ejecución  """)


query2 = "select * from ejecucion_mes"

##data2 = pd.read_sql_query(query2, conn)

data2 = pd.read_csv('df/ejecucion_mes.csv',delimiter=',',on_bad_lines='skip', encoding='ISO-8859-1')
data2['fecha']=pd.to_datetime(data2['fecha'])


AgGrid(data2,height=200)
st.divider()
st.subheader(""" 💹 Grafica de Ejecución   """)
col1, col2 = st.columns((2))
# Getting the min and max date 
startDate = pd.to_datetime(data2["fecha"]).min()
endDate = pd.to_datetime(data2["fecha"]).max()

with col1:
    date1 = pd.to_datetime(st.date_input("Fecha Inicio", startDate))

with col2:
    date2 = pd.to_datetime(st.date_input("Fecha Fin", endDate))


tipo_dato=st.selectbox('Seleccion el tipo', ['Todos','Inversion','Gasto','Recurso'],placeholder='Seleccione el tipo')
##quit()
data2['fecha'] = data2['fecha'].dt.tz_localize(None)
datos_filtrados = data2[(data2["fecha"] >= date1) & (data2["fecha"]<= date2)].copy()

chart_recurso=alt.Chart(datos_filtrados).mark_line(color='red').encode(
        x=alt.X('fecha:T',axis=alt.Axis(title='Fecha')),
        y=alt.Y('recurso:Q',axis=alt.Axis(title='Recurso')),
        tooltip=[alt.Tooltip('fecha:T', title='Fecha'), alt.Tooltip('recurso:Q', title='Recurso')]

    )

chart_gasto=alt.Chart(datos_filtrados).mark_line(color='blue').encode(
        x=alt.X('fecha:T',axis=alt.Axis(title='Fecha')),
        y=alt.Y('gasto:Q',axis=alt.Axis(title='Gasto')),
        tooltip=[alt.Tooltip('fecha:T', title='Fecha'), alt.Tooltip('gasto:Q', title='Monto')]

    )

chart_inversion=alt.Chart(datos_filtrados).mark_line(color='green').encode(
        x=alt.X('fecha:T',axis=alt.Axis(title='Fecha')),
        y=alt.Y('inversion:Q',axis=alt.Axis(title='Inversion')),
        tooltip=[alt.Tooltip('fecha:T', title='Fecha'), alt.Tooltip('inversion:Q', title='Monto')]

    )
if(tipo_dato=='Todos'):
    final_chart = alt.layer( chart_gasto, chart_recurso,chart_inversion).properties(
        width=600,
        height=300
    )
if(tipo_dato=='Inversion'):
   final_chart = alt.layer(chart_inversion).properties(
        width=600,
        height=300
    ) 
if(tipo_dato=='Gasto'):
    final_chart = alt.layer(chart_gasto).properties(
        width=600,
        height=300
    )
if(tipo_dato=='Recurso'):
    final_chart = alt.layer(chart_recurso).properties(
        width=600,
        height=300
    )  
st.altair_chart(final_chart,use_container_width=True)

t = pivot_ui(data2)
st.divider()
st.subheader("📉Tabla Dinámica de Datos")
# Muestra la tabla pivote en Streamlit
with open(t.src) as t:
    components.html(t.read(),  height=500, scrolling=True)

footer = """
<div style="text-align:center; padding: 10px; color: gray;">
     🖥️ Realizado por Rudy Manzaneda @2023
</div>
"""

st.markdown(footer, unsafe_allow_html=True)



def ocultar():
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

    hide_streamlit_style2 = """
                <style>
                .stDeployButton {display: none;}
                </style>
                """
    st.markdown(hide_streamlit_style2, unsafe_allow_html=True) 
ocultar()