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
import plotly.express as px
import calendar


database_params = {
    'host': 'localhost',
    'database': 'DBANALISISPRES',
    'user': 'postgres',
    'password': '123456',
    'port': '5432',  # Puerto predeterminado de PostgreSQL
}
conn = psycopg2.connect(**database_params)

# Ejecutar una consulta y cargar los resultados en un DataFrame de Pandas
query = "SELECT proyecto,financiador," \
"sum(g_2012) as g_2012, sum(g_2013) as g_2013, sum(g_2014) as g_2014, sum(g_2015) as g_2015,sum(g_2016) as g_2016,sum(g_2017) as g_2017,sum(g_2018) as g_2018,sum(g_2019) as g_2019," \
"sum(g_2020) as g_2020,sum(g_2021) as g_2021,sum(g_2022) as g_2022," \
"(sum(g_2012) +  sum(g_2013) +sum(g_2014) +sum(g_2015) +sum(g_2016) +sum(g_2017) +sum(g_2018) +sum(g_2019) +sum(g_2020) +sum(g_2021) +sum(g_2022)) as total" \
" FROM ejecucion_proyecto " \
" GROUP BY proyecto,financiador order by proyecto"

data = pd.read_sql_query(query, conn)
datos=data.copy()
st.set_page_config(page_title="Gasto Seguro - Datos Economicos por Ejecucion de proyecto", page_icon=":pie:",layout="wide")
st.write("""
""")


st.subheader(" 💰 CMI Ejecucion Proyecto")

st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
st.sidebar.subheader("💲 Seleccion el Filtro ")
proyectos = datos['proyecto'].unique()

proyectos_con_todos = ['Todos los Proyectos'] + list(proyectos)



# Mostrar el selectbox en la barra lateral
proyecto = st.sidebar.selectbox('Proyectos', proyectos_con_todos)

financiadores=datos[(datos['proyecto']==proyecto)]
datos_financiadores=financiadores['financiador'].unique()
financiadores_con_todos = ['Todos'] + list(datos_financiadores)
financiador = st.sidebar.selectbox('Financiador', financiadores_con_todos)

if(proyecto=='Todos los Proyectos' and financiador=='Todos' ):
    AgGrid(datos,height=200, width=500)
    filter_proyecto=datos.sum()
    filter_proyecto=filter_proyecto.transpose().reset_index()
    filter_proyecto.columns = ['gestion', 'monto_ejecucion']
    filter_proyecto=filter_proyecto.drop([0,1,13])
    df_transposed=filter_proyecto
else:
    filter_proyecto=datos[(datos['proyecto']==proyecto)]
    AgGrid(filter_proyecto,height=200, width=500)
    
    filter_proyecto=filter_proyecto.iloc[:, 2:13]
    df_transposed = filter_proyecto.transpose().reset_index()
    df_transposed.columns = ['gestion', 'monto_ejecucion']
# Mostrar la tabla utilizando AgGrid con opciones personalizadas
st.subheader("🏦 Proyecto:"+proyecto)
## ordenando datos
print(filter_proyecto)

# Utilizar pivot_table para pivotar los datos


# Mostrar el resultado
print(df_transposed)

chart2=alt.Chart(df_transposed).mark_line().encode(
    x=alt.X('gestion:N',axis=alt.Axis(title='Gestion')),
    y=alt.Y('monto_ejecucion:Q',axis=alt.Axis(title='Monto Ejecutado')),
    tooltip=[alt.Tooltip('gestion:N', title='Gestion'), alt.Tooltip('monto_ejecucion:Q', title='Monto')]

).properties(
    width=600,
    height=300
)
st.altair_chart(chart2,use_container_width=True)

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
