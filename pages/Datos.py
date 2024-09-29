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
# Parámetros de conexión
database_params = {
    'host': 'dpg-crsbjetds78s73e3d0f0-a.ohio-postgres.render.com',
    'database': 'gastoseguro',
    'user': 'rmanzanedav',
    'password': 'ZG7LFQBeEdQQxxZONAJCgV8Fcm2Y8ts1',
    'port': '5432',  # Puerto predeterminado de PostgreSQL
}
conn = psycopg2.connect(**database_params)
# Crear una conexión utilizando SQLAlchemy
##engine = create_engine(f'postgresql://{database_params["user"]}:{database_params["password"]}@{database_params["host"]}:{database_params["port"]}/{database_params["database"]}')

# Ejecutar una consulta y cargar los resultados en un DataFrame de Pandas
query = "select * from ejecucion order by proyecto,gestion_presupuestaria"

data = pd.read_sql_query(query, conn)
datos=data.copy()
st.set_page_config(page_title="Gasto Seguro - Datos Economicos de Ejecucion Publica", page_icon=":pie:",layout="wide")
st.write("""
""")
st.title("📈 GASTO SEGURO")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
st.sidebar.subheader("Seleccion el Filtro ")
gestion= st.sidebar.selectbox('Gestion',datos['gestion_presupuestaria'].unique())


datos= datos[datos['gestion_presupuestaria']==gestion]
print(datos["proyecto"].unique())
proyecto = st.sidebar.selectbox("Seleccione el Proyecto", datos["proyecto"].unique(),help="Elige una presupuesto.",placeholder="Elije un Proyecto")

if gestion is None:
    criterio=f" "
else:
    criterio=f"and  gestion_presupuestaria = '{gestion}'"
query2=f"select sum(monto_ejecutado) as monto,proyecto as proyectos from ejecucion  where proyecto<>'Administracion Central' {criterio} GROUP BY proyecto"
data2 = pd.read_sql_query(query2, conn)

chart2=alt.Chart(data2).mark_line().encode(
    x=alt.X('proyectos:N',axis=alt.Axis(title='Proyectos')),
    y=alt.Y('monto:Q',axis=alt.Axis(title='Monto')),
    tooltip=[alt.Tooltip('proyectos:N', title='Proyecto'), alt.Tooltip('monto:Q', title='Monto')]

).properties(
    width=600,
    height=300
)
st.subheader(f" 📋Datos de Ejecucion por proyectos de la Gestion:{gestion}")
st.altair_chart(chart2,use_container_width=True)
st.divider()
##datos= datos[datos['gestion_presupuestaria'].isin(gestion)]
##ppto =st.sidebar.selectbox('Seleccion el Presupuesto',list(datos['desc_presupuesto'].unique()),placeholder="Seleccione el Presupuesto")
def format_sales(value):
    if value >= 0:
        return 'Bs. {:.2f} '.format(value / 1_000_00)

data2["monto (Formatted)"] = data2["monto"].apply(format_sales)
fig4 = px.treemap(data2, path = ["proyectos","monto"], values = "monto",
                  hover_name = "monto (Formatted)",
                  hover_data = ["monto (Formatted)"],
                  color = "proyectos", height = 700, width = 600)
fig4.update_traces(textinfo="label+value")
st.subheader(" 📝 Grafica de Box correspondiente al Monto Total por ejecucion y Proyecto")
st.plotly_chart(fig4,use_container_width=True)
st.divider()

st.subheader(f"📈 Datos del Proyecto:{proyecto}")
data_mes=data[(data['proyecto']==proyecto) & (data['gestion_presupuestaria']==gestion)].copy()
mes=st.selectbox('Seleccione el Mes',data_mes['mes'].unique())
filtered_data_pie = data[(data['proyecto']==proyecto) & (data['gestion_presupuestaria']==gestion  )& (data['mes']==mes)]
fig5 = px.pie(filtered_data_pie, values = "monto_ejecutado", names = "nombre_partida", template = "plotly_dark")
fig5.update_traces(text = filtered_data_pie["nombre_partida"], textposition = "inside")
st.plotly_chart(fig5,use_container_width=True)

st.subheader(" 📈 Grafica Estadistica por partida y Gestion")
query4=(f"select sum(monto_ejecutado) as monto,proyecto as proyectos, nombre_partida, gestion_presupuestaria, periodo"
        f" from ejecucion"  
        f" where 0=0  {criterio}"
        f" GROUP BY proyecto, nombre_partida, gestion_presupuestaria,periodo" 
    )

data_filter_partida = pd.read_sql_query(query4, conn)
data_filter_partida=data_filter_partida[(data_filter_partida['proyectos']==proyecto)]

meses_dict_espanol = {i: calendar.month_name[i] for i in range(1, 13)}

# Supongamos que 'periodo' es una columna que contiene números de mes (1-12)
data_filter_partida['mes'] = data_filter_partida['periodo'].map(meses_dict_espanol)

orden_meses_espanol = [calendar.month_name[i] for i in range(1, 13)]
data_filter_partida['mes'] = pd.Categorical(data_filter_partida['mes'], categories=orden_meses_espanol, ordered=True)
data_filter_partida = data_filter_partida.sort_values('mes')


# Supongamos que 'nombre_partida' es la columna que quieres modificar
partidas_unicas = data_filter_partida['nombre_partida'].unique()

# Agregar 'Todas las Partidas' al inicio de la lista
partidas_unicas = ['Todas las Partidas'] + list(partidas_unicas)

# Crear un nuevo DataFrame con el registro agregado
nueva_fila = {'nombre_partida': 'Todas las Partidas'}
data_filter_partida = pd.concat([pd.DataFrame([nueva_fila]), data_filter_partida], ignore_index=True)



partida= st.selectbox('Partida',data_filter_partida['nombre_partida'].unique())
if(partida!='Todas las Partidas'):
    data_filter_partida=data_filter_partida[(data_filter_partida['nombre_partida']==partida)]


fig6 = px.bar(data_filter_partida, x = "mes", y = "monto", text = ['Bs{:,.2f}'.format(x) for x in data_filter_partida["monto"]],
            ## color="mes",  # Agrega el argumento color
            color_discrete_sequence=['orange'], 
              template="seaborn",
              title='Monto ejecutado por mes y partida',  # Título del gráfico
              )

st.plotly_chart(fig6,use_container_width=True, height = 200)


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
