import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from pandas import set_option
from matplotlib import pyplot
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
def datosDescriptivos(data):
    st.subheader('shape')
    st.write(data.shape)

    st.write('types')
    st.write(data.dtypes)
    ##st.write('descriptions')
    ##set_option('precision', 1)
    ##st.write(data.describe())
    # correlation
    st.write('correlation')
    st.write(data.corr(method='pearson'))
    # histograms
    st.write('histograms')
    data.hist(sharex=False, sharey=False, xlabelsize=1,
             ylabelsize=1)
    # Crear el histograma con Plotly Express
    fig = px.histogram(data, x='sbp', marginal='box', nbins=20)


    # Mostrar el histograma con Streamlit
    st.plotly_chart(fig)
    fig2 = px.box(data, points="all",color_discrete_sequence=['green', 'red'])
    st.plotly_chart(fig2)
    # Crear una matriz de dispersión con Plotly Express
    fig3 = px.scatter_matrix(data,color_continuous_scale='Viridis')

    # Mostrar la matriz de dispersión con Streamlit
    st.plotly_chart(fig3)
    fig4 = px.imshow(data.corr(), color_continuous_scale='Viridis')
    # Personalizar el diseño del gráfico para mostrar los valores
    fig4.update_layout(
        xaxis=dict(side="bottom"),
        width=600,
        height=600
    )

    # Mostrar la matriz de correlación con Streamlit
    st.plotly_chart(fig4)
def evaluarAlgoritmos(X_train, y_train, input):
    num_folds = 10
    seed = 7

    #scoring = 'neg_mean_squared_error'
    scoring = 'r2'

    # Spot-Check Algorithms
    pipelines = []
    pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])))
    pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
    pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
    pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
    pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
    pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
    pipelines.append(('ScaledMLP', Pipeline([('Scaler', StandardScaler()),('MLP', MLPRegressor(max_iter=2000))])))
    pipelines.append(('ScaledMLP', Pipeline([('Scaler', StandardScaler()),('RFC', RandomForestClassifier())])))

    results = []
    names = ['LR','LASSO','EN','KNN','CART','SVR','MLP','RFC']
    msg=[]
    for name, model in pipelines:
        kfold = KFold(n_splits=num_folds)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg.append( "%s R^2: %.3f (%.3f)" % (name, cv_results.mean(), cv_results.std()))
    st.write(pd.DataFrame(msg))
    
    ##fig8 = px.box(x=names, y=results, points="all", title='Comparación de Algoritmos con Datos Estandarizados')
    ##fig8.update_layout(xaxis_title='Algoritmo', yaxis_title='MSE')
    ##st.plotly_chart(fig8)


encoder= LabelEncoder()
st.set_page_config(page_title="App de Prediccion",
                   page_icon="",
                   layout="centered",
                   initial_sidebar_state="auto"
                   )

st.title("App de predicción de enfermedades cardiacas")
st.markdown("""---""")

##logo= "imagen.jpg"
##st.sidebar.image(logo, width=100)
st.sidebar.header("Datos Ingresados por el usuario")

upload_file=st.sidebar.file_uploader("Cargue sus datos en CSV", type=["csv"])

if upload_file is not None:
    input_df=pd.read_csv(upload_file)
    
else:
    def user_input_features():
        sbp=st.sidebar.slider('Presion Arterial Sistolica ',101,218,150)
        Tabaco= st.sidebar.slider('Tabaco Acumulado (Kg) ',0.00,31.20,2.00)
        ldl=st.sidebar.slider('Colesterol de Lipoproteinas de Baja Densidad ',0.98,15.33,4.34)
        Adiposidad=st.sidebar.slider('Adiposidad',6.74,42.49,2.12)
        Familia=st.sidebar.slider('Antecedentes Familiares de Enfermedad Cardiaca',0,1)
        Tipo= st.sidebar.slider('Tipo',13,78,53)
        Obesidad= st.sidebar.slider('Obesidad',14.70,46.58,25.80)
        Alcohol= st.sidebar.slider('Consumo Actual de Alcohol',0.00,147.19,7.51)
        Edad= st.sidebar.slider('Edad',15,64,45)

        data={
            'sbp':sbp,
            'Tabaco':Tabaco,
            'ldl':ldl,
            'Adiposidad':Adiposidad,
            'Familia':Familia,
            'Tipo':Tipo,
            'Obesidad':Obesidad,
            'Alcohol':Alcohol,
            'Edad':Edad
        }

        features= pd.DataFrame(data, index=[0])
        ##print(features)
        return features
    input_df=user_input_features()

encoder=LabelEncoder()
input_df['Familia']=encoder.fit_transform(input_df['Familia'])
#Seleccionas solo la primera fila
input_df=input_df[:1]

st.subheader('Datos Ingresados por el usuario')

if upload_file is not None:
    st.write(input_df)
else:
    st.write('A la espera de que se cargue el archivo csv. con parametros de ejemplo')
    st.write(input_df)



data= pd.read_csv('heart.csv')
datosDescriptivos(data)

data['Familia']=encoder.fit_transform(data['Familia'])

X= data.drop('chd', axis=1)
print(X)
y=data['chd']
print(y)


X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.3, random_state=0)


evaluarAlgoritmos(X_train, y_train,input_df)



from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

clf.fit(X_train,y_train)

prediction=clf.predict(input_df)
prediction_proba=clf.predict_proba(input_df)
##load_clf= pickle.load(open('heart.pkl','rb'))


##prediction= load_clf.predict(input_df)
##prediction_proba= load_clf.predict_proba(input_df)

col1,col2= st.columns(2)

with col1:
    st.subheader('Prediccion')
    st.write(prediction)
with col2:
    st.subheader('Probabilidad de prediccion')
    st.write(prediction_proba)
st.markdown("""------""")