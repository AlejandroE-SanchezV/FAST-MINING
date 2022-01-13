# MINERÍA DE DATOS - PROYECTO FINAL
#Alumno: Sánchez Vázquez Alejandro Enrique



import streamlit as st
import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
import io                         #Canalización de la salida de DataFrame.info() al búfer en lugar de sys.stdout - Para creación de contenido en el búfer y su respectiva escritura en un archivo de texto
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import random                   #para generació de colores aleatorios
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler    #Para realziar estandarización de la data
from sklearn.decomposition import PCA               #Para implementación del módulo de Análisis de Componentes Principales

#Bibliotecas para la implementación del módulo de clusterización (Clustering Jerárquico Ascendente)
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

#Bibliotecas para la implementación del módulo de clusterización (Clustering Particional)
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

#Biblioteca para implementar el método del codo en algoritmo de clusterización particional
from kneed import KneeLocator

#Biblioteca para implementar el componente de reglas de asociación mediante algoritmo A priori
from apyori import apriori

#from progressbar import ProgressBar     #Para intento de progress bar
#https://www.analyticsvidhya.com/blog/2021/06/generate-reports-using-pandas-profiling-deploy-using-streamlit/


#Biblioteca para implementar el componente de reglas de árboles de decisión (pronóstico)


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection
#Para visualziación del árbol
import graphviz
from sklearn.tree import export_graphviz
import lightgbm as lgb

#https://discuss.streamlit.io/t/graphviz-plot/1042/7
#sin funcionar

#Forma alternativa de visualización
from sklearn.tree import plot_tree
from sklearn.tree import export_text



#Bibliotecas para árboles de clasificación
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection




from PIL import Image

st.set_page_config(
     page_title="FAST MINING",
     page_icon="rayo.png",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get help': 'https://docs.streamlit.io',
         'Report a bug': "https://discuss.streamlit.io",
         'About': "Este proyecto fue creado como proyecto final de la asignatura de Minería de Datos por Alejandro Enrique Sánchez Vázquez, alumno de noveno semestre de Ingeniería Mecatrónica de la  Facultad de Ingeniería de la UNAM."
     }
 )


image = Image.open('fastmining.jpg')

st.sidebar.image(image)


@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')

def explore(data):
    df_types = pd.DataFrame(data.dtypes, columns=['Data Type'])
    numerical_cols = df_types[~df_types['Data Type'].isin(['object','bool'])].index.values
    df_types['Count'] = data.count()
    df_types['Unique Values'] = data.nunique()
    #df_types['Min'] = data[numerical_cols].min()
    #df_types['Max'] = data[numerical_cols].max()
    #df_types['Average'] = data[numerical_cols].mean()
    #df_types['Median'] = data[numerical_cols].median()
    #df_types['St. Dev.'] = data[numerical_cols].std()
    return df_types.astype(str)
    #Solución a dataframe.dtypes
    #https://discuss.streamlit.io/t/streamlitapiexception-unable-to-convert-numpy-dtype-to-pyarrow-datatype/18253


#FUNCIÓN PARA LIMPIAR EL DATAFRAME
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)



#algoritmos0=['Home']


st.sidebar.title("FAST MINING")

st.sidebar.write("A través del uso de esta página web podrás analizar conjuntos de datos en formato CSV con diferentes algoritmos de Machine Learning.")

st.sidebar.write("Selecciona el componente con el cual deseas trabajar, y posteriormente se te desplegará el menú de los algoritmos disponibles.")

st.sidebar.write("\n")

#st.sidebar.subheader("Componente #1")

#st.sidebar.subheader("Componente #2")


options=['Análisis Exploratorio de Datos (EDA)','Selección De Características','Clusterización','Reglas de Asociación','Árboles de decisión']

componente= st.sidebar.selectbox("Selecciona el componente a utilizar", options, index=0, key=1000, help="Después de seleccionar un componente, se desplegárán las opciones de los algoritmos disponibles.", on_change=None, args=None, kwargs=None)


#st.markdown("![Alt Text](https://media.giphy.com/media/pOEbLRT4SwD35IELiQ/giphy.gif)")
#st.markdown("![Alt Text](https://media.giphy.com/media/lbcLMX9B6sTsGjUmS3/giphy.gif)")


image2 = Image.open('bd2.jpg')

st.image(image2)







#st.title("Minería de datos - Proyecto Final")
st.title("FAST MINING")
st.info("Bienvenid@ a esta plataforma de Minería de Datos con Inteligencia Artificial creada por Alejandro Enrique Sánchez Vázquez.")


if componente == 'Análisis Exploratorio de Datos (EDA)':
    algoritmos1=['EDA', 'EDA Avanzado']


    menu1 = st.sidebar.radio("Realiza la selección del tipo de Análisis Exploratorio de Datos (EDA) que deseas realizar: Básico o Avanzado", (algoritmos1))

    if menu1 == "EDA":

        st.title("Análisis Exploratorio de Datos")
        st.subheader("Carga del dataset y visualización del dataframe")

        dataset=st.file_uploader("Introduce tu archivo CSV para realizar el EDA", type='csv', accept_multiple_files=False, key=1, help="Sólo se permite la carga de un sólo archivo en formato CSV",on_change = None, args = None, kwargs = None)
        if dataset is not None:
            st.write("")
            dataframe = pd.read_csv(dataset)
            st.subheader("Visualización del dataframe generado con el archivo cargado")
            st.dataframe(dataframe)

            
            st.subheader("Número de filas y columnas del dataframe")
            #Obtención del número de filas del dataframe
            filas=dataframe.index
            num_filas=(len(filas))

            #Obtención del número de columnas del dataframe
            colDF=dataframe.columns
            num_columnas=len(colDF)

            st.write("El dataframe cargado tiene", num_filas, "filas y ", num_columnas, "columnas")


            st.subheader("Visualización del dataframe a partir de variables específicas")
            columns_names = dataframe.columns.values

            #st.selectbox("Selecciona las variables específicas que deseas visualizar en este dataframe personalizado", columns_names , index=0, help=None, on_change=None, args=None, kwargs=None)
            variables_seleccionadas_EDA=st.multiselect("Selecciona las variables específicas que deseas visualizar en este dataframe personalizado", columns_names, default=dataframe.columns.values[0], key=1, help=None, on_change=None, args=None, kwargs=None)


            #VALIDACIÓN DE LAS COLUMNAS SELECCIONADAS EN EL MULTISELECT Y EL TIPO DE DATO EN EL QUE SE GUARDAN
            #st.write(variables_seleccionadas_EDA) 
            #st.write(type(variables_seleccionadas_EDA))
            #----------------------------------------------------------------------

            #INTENTO FALLLIDO DE HABILITAR EL DESPLIEGE DEL DATAFRAME DE OTRO MODO
            #filtro=columns_names.isin(variables_seleccionadas_EDA)
            #EDAdataframe_filtrado=dataframe[filtro]
            #stringEDAdataframe = "".join(variables_seleccionadas_EDA)
            
            #Validación de elemntos seleccionados en el Multiselect Input Widget 
            #st.write(arrayEDAdataframe)------------------------------------
            
            #Creación de array a partir de la lista con los items proveientes del multiselect
            arrayEDAdataframe = np.array(variables_seleccionadas_EDA)
            st.dataframe(dataframe[arrayEDAdataframe])


            #st.write(num_filas)
            #Obtención de sólo las primeras n columnas de dataframe, donde n es el dato que va dentro de los paréntesis.
            st.subheader("Visualización de las primeras n filas del dataframe")
            st.caption("A través del slider se permite la visualización de las primeras n filas. Seleccione el número n a continuación.")
            num_filas_seleccionadas=st.slider("Número n", min_value=1, max_value=num_filas, value=None, step=1, format=None, key=1, help="Seleccione un número entero para poder ver el dataframe desde la fila uno hasta la fila N.", on_change=None, args=None, kwargs=None)
            st.dataframe(dataframe.head(num_filas_seleccionadas))


            #CREACIÓN DE FILTRO AVANZADO
            st.subheader("Filtro avanzado")
            st.caption("Utiliza el siguiente widget y slider para combinar la funcionalidad de las 2 opciones anteriores")

            #SECCIÓN EN DOS COLUMNAS
            col1,col2=st.columns(2)

            with col1:    #Columna 1 para selección de variables 
                variables_seleccionadas_EDA_Bavanzada=st.multiselect("Variables específicas a visualizar en la búsqueda avanzada", columns_names, default=dataframe.columns.values[0], key=3, help=None)
                arrayEDAdataframe_Avanzada = np.array(variables_seleccionadas_EDA_Bavanzada)


            with col2:     #Columna 2 para selección de número de filas a mostrar
                num_filas_seleccionadas_Bavanzada=st.slider("Selecciona el número n de filas a mostrar", min_value=1, max_value=num_filas, value=None, step=1, format=None, key=2, help="Seleccione un número entero para poder ver el dataframe desde la fila uno hasta la fila N.", on_change=None, args=None, kwargs=None)

            st.dataframe(dataframe.head(num_filas_seleccionadas_Bavanzada)[arrayEDAdataframe_Avanzada])

            
            st.subheader("Descripción de la estructura de los datos")
            st.write(explore(dataframe))

            st.subheader("Identificación de datos faltantes")
            st.caption("En esta sección podrás revisar la suma de todos los valores no nulos de cada variable, el tipo de dato que representan, y el uso de memoria.")
            
            #Intento de cambiar nombre de la primera columna
            #df_new = dataframe.rename(columns={'0': 'data'})
            #https://note.nkmk.me/en/python-pandas-dataframe-rename/



            #st.write(dataframe.info())
            #https://discuss.streamlit.io/t/df-info-is-not-displaying-any-result-in-st-write/5100
            
            buffer = io.StringIO()
            dataframe.info(buf=buffer)
            s = buffer.getvalue()

            st.text(s)

            #Posible implementación de seleccionar variable única en específico, e.g. un país en dataset de registro de vacunación por país
            #https://discuss.streamlit.io/t/select-an-item-from-multiselect-on-the-sidebar/1276


            st.header("Detección de valores atípicos")
            st.write("Generación de gráficos para tener una idea general de las distribuciones de los datos")


            #Gráficos con MATPLOTLIB
            fig0, ax = plt.subplots(figsize=(18,18))
            dataframe.hist(figsize=(18,18), xrot=45, ax=ax)
            st.pyplot(fig0)




            st.subheader("Resumen Estadístico de Variables Numéricas")
            st.caption("En este apartado podrás apreciar las estadísticas más reperesentativas de cada variable numérica, tales como el recuento del número de registros, el promedio, la desviación estandar, el mínimo y máximo, así como también la información de los cuartiles del dataset.")
            st.write(dataframe.describe())

            st.subheader("Diagramas para detectar posibles valores atípicos")
            st.caption("En esta sección podrás visualizar a través de diagramas de caja de tu variable de selección los posibles valores atípicos a través de la generación de diagramas de caja.")


            numerical_columns=dataframe.select_dtypes(include=np.number).columns.tolist()
            #numerical_columns=dataframe.select_dtypes(include=np.number).columns.tolist()
            #t.write(numerical_columns)

            variables_Valores_Atipicos=st.multiselect("Selecciona las variables específicas que deseas visualizar en los diagramas de caja.", numerical_columns, default=numerical_columns[1], key=3, help=None, on_change=None, args=None, kwargs=None)



            #Gráficos con Seaborn
            
            for col in variables_Valores_Atipicos:
                graph_2 ,ax= plt.subplots(figsize=(8, 4.5))
                boxplots=sns.boxplot(col, data=dataframe, ax=ax)
                st.pyplot(graph_2)

            

            #Solución de Error de Identación
            #Ctrl+Shift+P or View->Command Palette
            #Convert Indentation to Spaces

            #https://stackoverflow.com/questions/5685406/inconsistent-use-of-tabs-and-spaces-in-indentation

            st.header("Distribución de Variables Categóricas")
            

            #Intento de despliegue de la función describe() con variables categóricas (errores de streamlit debido a actualización)
            #https://discuss.streamlit.io/t/after-upgrade-to-the-latest-version-now-this-error-id-showing-up-arrowinvalid/15794/6

            #Creación de datafrane específico sólo para variables categóricas
            
            #array_Object_Variables_EDA = np.array(columns_object)
            #dataframe_objectsEDA=st.dataframe(dataframe[array_Object_Variables_EDA].describe(include='all'))
            #st.write(dataframe_objectsEDA.describe())
            
            #dataframe_objectsEDA.dtypes.astype(str)
            #st.write("Características nominales",dataframe_objectsEDA.describe(include='object'))

            #if st.button("Presiona este si tu dataset incluye variables categóricas", key=3, help="La variables categóricas son aquellas cuyos datos están formados por valores no numéricos", on_click=None, args=None, kwargs=None):
            
            col_names=dataframe.select_dtypes(include='object').columns.tolist()

            if len(col_names)==0:
                    st.warning("El conjunto de datos no tiene variables categóricas, por lo cual no se puede hacer uso de esta función")

            else:
                st.subheader("Histogramas para variables categóricas")
                col_names=dataframe.select_dtypes(include='object').columns.tolist()

                #st.write(col_names)

                    #variables_edaObjects=st.multiselect("Selecciona las variables categóricas de las cuales deseas visualizar su histograma.", col_names, default=col_names[2], key=5, help=None, on_change=None, args=None, kwargs=None)
                variables_edaObjects = st.selectbox("Selecciona la variable categóricas de la cual deseas visualizar su histograma.", col_names, index=0, key=3, help=None, on_change=None, args=None, kwargs=None)
                st.write(variables_edaObjects)

                    #objectCols=dataframe.select_dtypes(include='object')
                    #st.write(objectCols)

                graph_3 ,ax= plt.subplots(figsize=(8, 4.5))
                dataframe[variables_edaObjects].value_counts().plot(kind='bar');
                st.pyplot(graph_3)


                st.subheader("Agrupación por variables categóricas")
                st.caption("En esta sección se puede apreciar el promedio de cada una de las variables numéroicas respecto a cada una de las variables categóricas del dataframe")

            
                if st.button("Presiona este botón en dado caso que requieras de esta funcionalidad (sólo si tienes variables categóricas en tu dataset)", key=2, help=None, on_click=None, args=None, kwargs=None):

                    for col in dataframe.select_dtypes(include='object'):
                        st.write((dataframe.groupby(col).agg(['mean'])))

                else:
                    st.write("")

            #else:
                #st.write("Si tu dataset sólo contiene valores numéricos, haz caso omiso a esta sección, en caso contrario se obtendrá un error como resultado debido a la falta de ese tipo de variables en el conjunto de datos. Si presionaste el botón por error, trata de utilizar de alguuna funcionalidad de las de arriba y después vuelve a esta parte si deseas continuar con el EDA para visualizar la relación entre pares de variables más adelante.")

            

            #for col in dataframe.select_dtypes(include='object'):
                #if dataframe[col].nunique()<15:sns.countplot(y=col, data=dataframe)
                #graph_3 ,ax= plt.subplots(figsize=(8, 4.5))
                #dataframe.value_counts().plot(kind='bar');
                #st.pyplot(graph_3)



            
            #r = random.random()
            #b = random.random()
            #g = random.random()
            #dataframe[variables_edaObjects].value_counts().plot(kind='bar', color=(r,g,b));



           




            st.header("Identificación de relaciones entre pares variables")
            st.subheader("Matríz de correlaciones")

            st.write(dataframe.corr())

            st.subheader("Mapa de Calor")



            graph_4 ,ax= plt.subplots(figsize=(14, 7))
            sns.heatmap(dataframe.corr(), cmap='RdBu_r', annot=True)
            st.pyplot(graph_4)





    elif menu1 == "EDA Avanzado":

        #https://www.analyticsvidhya.com/blog/2021/06/generate-reports-using-pandas-profiling-deploy-using-streamlit/
        #https://discuss.streamlit.io/t/including-pandas-profiling-report-in-streamlit/473/6
        #https://github.com/ashishgopal1414/streamlit-pandas-profiling
        #profile = ProfileReport(dataframe, explorative=True)

        #st.write(profile.html, unsafe_allow_html = True)

        st.title("Análisis Exploratorio de Datos Avanzado")
        dataset_EDA_Avanzado=st.file_uploader("Introduce tu archivo CSV para realizar el Análisis Exploratorio De Datos Avanzado", type='csv', accept_multiple_files=False, key=2, help="Sólo se permite la carga de un sólo archivo en formato CSV",on_change = None, args = None, kwargs = None)
        
        if dataset_EDA_Avanzado is not None:
            st.write("")
            dataframe_Advanced = pd.read_csv(dataset_EDA_Avanzado)
            st.subheader("Visualización del dataframe generado con el archivo cargado")
            st.dataframe(dataframe_Advanced)

            st.subheader("Análisis Avanzado del perfil del conjunto de datos")

            profile2 = ProfileReport(dataframe_Advanced, explorative=True)

            with st.spinner(text="En progreso (esto puede tardar algunos segundos)..."):
                
                #def job():
                    #bar = progressbar.ProgressBar()
                    #for i in bar(range(100)):
                        #st_profile_report(profile2)
                        #time.sleep(0.01)

                #job()    #Intento de implementación de progressbar

                st_profile_report(profile2)

            st.success('¡Listo!')






elif componente == "Selección De Características":
    algoritmos2=['Análisis Correlacional de Datos', 'Análisis de Componentes Principales']


    menu2= st.sidebar.radio("Realiza la selección del tipo de Selección de Características que deseas llevar a cabo.", (algoritmos2))


    if menu2 == "Análisis Correlacional de Datos":

        st.title("Análisis Correlacional de Datos")

        #Botón para subir archivo CSV

        dataset=st.file_uploader("Introduce tu archivo CSV para realizar el Análisis Correlacional de Datos", type='csv', accept_multiple_files=False, key=3, help="Sólo se permite la carga de un sólo archivo en formato CSV",on_change = None, args = None, kwargs = None)
        if dataset is not None:
            st.write("")
            dataframe = pd.read_csv(dataset)
            st.subheader("Visualización del dataframe generado con el archivo cargado")
            st.dataframe(dataframe)

            
            st.subheader("Número de filas y columnas del dataframe")
            #Obtención del número de filas del dataframe
            filas=dataframe.index
            num_filas=(len(filas))

            #Obtención del número de columnas del dataframe
            colDF=dataframe.columns
            num_columnas=len(colDF)

            st.write("El dataframe cargado tiene", num_filas, "filas y ", num_columnas, "columnas")

            st.subheader("Descripción de la estructura de los datos")
            st.write(explore(dataframe))

            st.subheader("Identificación de Datos Faltantes")
            st.write("En esta sección podrás ver el número de datos faltantes por cada variable del dataset.")
            st.write(dataframe.isnull().sum())







            st.header("Evaluación visual")
            col_names=dataframe.select_dtypes(include='object').columns.tolist()
            st.subheader("Selección de la variable categórica.")
            variables_Objects = st.selectbox("Selecciona la variable categórica de la cual deseas visualizar sus gráficas de dispersión entre pares de variables numéricas.", col_names, index=0, key=20, help=None, on_change=None, args=None, kwargs=None)
            if st.button("Presiona este botón en dado caso que requieras de esta funcionalidad (sólo si tienes variables categóricas en tu dataset)", key=20, help=None, on_click=None, args=None, kwargs=None):

                #if len(col_names)==0:

                    #st.warning("El conjunto de datos no tiene variables categóricas, por lo cual no se puede hacer uso de esta función")

                if len(col_names) !=0:

                    

                    #Conversión de la cadena proveniente del select a un array para poder usarlo como variable en hue 
                        
                    with st.spinner("Esto podría tardar algunos minutos, por favor espera, o en caso contrario, puedes cancelar la creación de la gráfica sin afectar los siguientes apartados."):

                        
                
                        string=variables_Objects
                        arr= string.split()
                        #st.write(arr[0])

                        #graph_4 ,ax= plt.subplots(figsize=(14, 7))
                        x = dataframe
                        fig = sns.pairplot(x, hue=arr[0])
                        st.pyplot(fig)

                    st.success('Gráfica de dispersión entre pares de variables realizada con éxito')

                else:
                    st.warning("El conjunto de datos no tiene variables categóricas, por lo cual no se puede hacer uso de esta función")


            else:

                st.write("")

            st.subheader("Gráficas de dispersión entre pares de variables numéricas")
            st.write(explore(dataframe))


            Num_col_names=dataframe.select_dtypes(include = ["int16", "int32", "int64", "float16", "float32", "float64"]).columns.tolist()
            num_var1= st.selectbox("Selecciona la primera variable numérica que tendrá el gráfico de dispersión.", Num_col_names, index=0, key=21, help=None, on_change=None, args=None, kwargs=None)
            num_var2= st.selectbox("Selecciona la segunda variable numérica que tendrá el gráfico de dispersión.", Num_col_names, index=1, key=22, help=None, on_change=None, args=None, kwargs=None)


            #str1=num_var1
            #str2=num_var2

            #arr1= str1.split()
            #arr2= str2.split()

            #st.write(arr1[0])
            #st.write(arr2[0])

            graph_5 ,ax= plt.subplots(figsize=(14, 7))
            plt.plot(dataframe[num_var1], dataframe[num_var2], 'o')
            plt.title('Gráfico de dispersión')
            plt.xlabel(num_var1)
            plt.ylabel(num_var2)
            st.pyplot(graph_5)

            st.subheader("Gráficas de dispersión con variable categórica entre pares de variables numéricas")

            st.subheader("Selección de la variable categórica.")
            variables_Objects2 = st.selectbox("Selecciona la variable categórica de la cual deseas visualizar sus gráficas de dispersión entre pares de variables numéricas.", col_names, index=0, key=23, help=None, on_change=None, args=None, kwargs=None)
            st.subheader("Selección del par de variables numéricas.")
            num_var3 = st.selectbox("Selecciona la primera variable numérica que tendrá el gráfico de dispersión.", Num_col_names, index=0, key=25, help=None, on_change=None, args=None, kwargs=None)
            num_var4 = st.selectbox("Selecciona la segunda variable numérica que tendrá el gráfico de dispersión.", Num_col_names, index=1, key=26, help=None, on_change=None, args=None, kwargs=None)

            if st.button("Presiona este botón en dado caso que requieras de esta funcionalidad (sólo si tienes variables categóricas en tu dataset)", key=24, help=None, on_click=None, args=None, kwargs=None):

                

                    #st.warning("El conjunto de datos no tiene variables categóricas, por lo cual no se puede hacer uso de esta función")

                if len(col_names) !=0:

                    

                    #Conversión de la cadena proveniente del select a un array para poder usarlo como variable en hue 
                        
                    with st.spinner("Espera un momento, por favor..."):

                        
                
                        string=variables_Objects2
                        arr2= string.split()
                        #st.write(arr[0])

                        #graph_4 ,ax= plt.subplots(figsize=(14, 7))
                        #data = dataframe
                        fig22 ,ax= plt.subplots(figsize=(20, 12))
                        sns.scatterplot(x=num_var3, y=num_var4, data=dataframe, hue=arr2[0])
                        plt.title('Gráfico de dispersión')
                        plt.xlabel(num_var3)
                        plt.ylabel(num_var4)
                        st.pyplot(fig22)

                    st.success('Gráfica de dispersión entre pares de variables realizada con éxito')

                else:
                    st.warning("El conjunto de datos no tiene variables categóricas, por lo cual no se puede hacer uso de esta función")


            else:

                st.write("")

            

            st.header("Identificación de Relaciones entre Variables")
            st.write("Matriz de correlaciones para analizar la relación entre las variables numéricas:")
            st.write(dataframe.corr())

            st.subheader("Análisis de la correlación que tiene una variable en específico con las demás.")

            st.write("Selecciona la variable numérica de la cual deseas realizar un análisis específico de correlación")
            num_var5= st.selectbox("Selecciona la segunda variable numérica que tendrá el gráfico de dispersión.", Num_col_names, index=2, key=28, help=None, on_change=None, args=None, kwargs=None)

            st.write(dataframe.corr()[num_var5].sort_values(ascending=False)[:10], '\n')

            st.subheader("Heatmap completo para el Análisis Correlacional de Datos")

            fig23 ,ax= plt.subplots(figsize=(20, 12))
            sns.heatmap(dataframe.corr(), cmap='RdBu_r', annot=True)
            st.pyplot(fig23)

            optionsMap=['Parte Superior del Heatmap','Parte Inferior del Heatmap']

            st.subheader("Visualización de la parte superior o inferior del Heatmap")
            st.write("Si sólo deseas visualiar la parte superior o inferior del heatmap para un mejor análisis, selecciona a través del siguiente menú la parte específica del mapa que quieres visualizar")
            heatmap_option= st.selectbox("Selecciona la segunda variable numérica que tendrá el gráfico de dispersión.", optionsMap, index=1, key=30, help=None, on_change=None, args=None, kwargs=None)

            if st.button("Haz click en este botón.", key=29, help=None, on_click=None, args=None, kwargs=None):

                
                if heatmap_option == 'Parte Superior del Heatmap':
                    
                    st.write("Parte Superior del Mapa de Calor:")


                    fig24 ,ax= plt.subplots(figsize=(20, 12))
                    MatrizSup = np.tril(dataframe.corr())
                    sns.heatmap(dataframe.corr(), cmap='RdBu_r', annot=True, mask=MatrizSup)
                    st.pyplot(fig24)

                elif heatmap_option == 'Parte Inferior del Heatmap':
                    
                    st.write("Parte Inferior del Mapa de Calor:")


                    fig25 ,ax= plt.subplots(figsize=(20, 12))
                    MatrizSup = np.triu(dataframe.corr())
                    sns.heatmap(dataframe.corr(), cmap='RdBu_r', annot=True, mask=MatrizSup)
                    st.pyplot(fig25)


            else:
                st.write("")


            st.subheader("Selección de características (Elección de variables)")
            st.write("Después de haber utilizado esta herramienta de Análisis Correlacional de Datos,, es hora de elegir qué variables deseas excluir del conjunto de datos con el fin de reducir la dimensionalidad y hacer un análisis de datos más eficiente.")
           
            columns_names = dataframe.columns.values
            st.write(columns_names)
            st.write()
            variables_eliminadas=st.multiselect("Selecciona las variables que deseas eliminar del análisis del dataframe", columns_names, key=31, help=None, on_change=None, args=None, kwargs=None)
            #st.write(variables_eliminadas)

            #dataframe.index = [""] * len(dataframe)
            #st.table(dataframe)

            #dataframe.assign: st.dataframe(dataframe.assign(hack='').set_index('hack'))
            #st.dataframe(dataframe)
            
            dataframe_ACD_actualizado=dataframe.drop(columns=variables_eliminadas)


            #styler = dataframe.style.hide_index()
            #st.write(styler.to_html(), unsafe_allow_html=True)

            


            st.dataframe(dataframe_ACD_actualizado)


            csv = convert_df(dataframe_ACD_actualizado)

            st.download_button(
               "Haz click aquí para descargar tu nuevo dataframe",
               csv,
               file_name="DataframeActualizado.csv",
               

               key='download-csv'
            )










        #Intento de despluegue de gráficas de dispersión con Seaborn
        
        #graph_5 ,ax= plt.subplots(figsize=(20, 12))
        #penguins = sns.load_dataset("penguins")
        #sns.pairplot(dataframe, hue=dataframe['Type'])
        #st.pyplot(graph_5)
        
        #object=st.selectbox
        

        #Despliegue de gráficas de dispersión entre pares de variables
        


        #Conversión de la cadena proveniente del select a un array para poder usarlo como variable en hue 
        #arr= string.split()
        #st.write(arr)

        #datavar = dataframe
        #fig = sns.pairplot(datavar, hue=arr[0])
        #st.pyplot(fig)


    elif menu2 == "Análisis de Componentes Principales":
        st.title("Análisis de Componentes Principales")


        dataset=st.file_uploader("Introduce tu archivo CSV para realizar el Análisis de Componentes Principales", type='csv', accept_multiple_files=False, key=4, help="Sólo se permite la carga de un sólo archivo en formato CSV",on_change = None, args = None, kwargs = None)
        if dataset is not None:
            st.write("")
            dataframe = pd.read_csv(dataset)
            st.subheader("Visualización del dataframe generado con el archivo cargado")
            st.dataframe(dataframe)

            
            st.subheader("Número de filas y columnas del dataframe")
            #Obtención del número de filas del dataframe
            filas=dataframe.index
            num_filas=(len(filas))

            #Obtención del número de columnas del dataframe
            colDF=dataframe.columns
            num_columnas=len(colDF)

            st.write("El dataframe cargado tiene", num_filas, "filas y ", num_columnas, "columnas")


            st.header("Estandarización de los datos")
            st.write("A continuación se muestra el dataframe actualizado que se utilizará para la implementación del Análisis de Componentes Principales, en el cual se incluyen exclusivamente a las variables numéricas, así como también se hace una limpieza del dataset.")
            normalizar = StandardScaler()                                         # Se instancia el objeto StandardScaler 
            col_names=dataframe.select_dtypes(include='object').columns.tolist()
            NuevaMatriz = dataframe.drop(columns=col_names)    # Se quitan las variables no necesarias (nominales)
            st.write(clean_dataset(NuevaMatriz))
            normalizar.fit(NuevaMatriz)                                           # Se calcula la media y desviación para cada variable
            MNormalizada = normalizar.transform(NuevaMatriz) 
            MNormalizada = pd.DataFrame(MNormalizada, columns=NuevaMatriz.columns)                   # Se normalizan los datos 

            st.subheader("Matriz Normalizada")

            st.write("A continuación se muestra la matriz con el conjunto de datos normalizado de las variables numericas.")
            clean_dataset(MNormalizada)
            st.write(MNormalizada)


            #Obtención del número de filas del dataframe
            filas_normalizadas=MNormalizada.index
            num_filasNorm=(len(filas_normalizadas))

            #Obtención del número de columnas del dataframe
            colDF_Norm=MNormalizada.columns
            num_columnas_Norm=len(colDF_Norm)

            st.write("En esta pequeña sección podrás aprecial el número de filas y columnas de la matriz normalizada.")
            col3, col4 = st.columns(2)
            col3.metric('Número de filas después de la normalización', num_filasNorm)
            sumaVarianza= num_columnas_Norm
            col4.metric('Número de columnas después de la normalización', sumaVarianza)


            #st.write(MNormalizada.shape)

            st.header("Cálculo de la matriz de covarianzas (matriz de correlaciones)")

            pca = PCA(n_components=None)           # Se instancia el objeto PCA, pca=PCA(n_components=None), pca=PCA(.85)
            pca.fit(MNormalizada)                  # Se obtiene los componentes
            st.write(pca.components_)

            st.header("Número de Componentes Principales")

            st.subheader("Proporción de varianza.")
            st.write("")
            Varianza = pca.explained_variance_ratio_
            st.write(Varianza)

            st.subheader("Varianza Acomulada")
            num_componentes=len(Varianza)

            #st.write(num_componentes)
            componentes = st.slider('Selecciona el número de los primeros n componentes quieres utilizar para realizar el cálculo de la varianza acomulada:', 
                min_value=1, 
                max_value=num_componentes, 
                value=None, 
                step=1, 
                format=None, 
                key='Slider_Var_Acomulada', 
                help='Este slider te muestra el número de componentes que has seleccionado para realizar la suma de la varianza de los mismos.', on_change=None, args=None, kwargs=None)
            
            st.write("Se recomienda seleccionar sólo hasta el número n de componentes que den del 75 al 90% porciento del total de la varianza")
            #st.write(componentes)
            sumaVarianza= sum(Varianza[0:componentes])

            st.subheader('Status de la varianza Acomulada')
            col1, col2 = st.columns(2)
            col1.metric('Número de componentes', componentes)
            sumaVarianza= (round(sumaVarianza,5))*100
            col2.metric('Porcentaje total de varianza acumulada', sumaVarianza)

            st.write('Actualmente has seleccionado los primeros', componentes, ' componentes, los cuales acomulan un porcentaje de varianza total del', sumaVarianza)

            st.subheader("Gráfica del número de componentes y de la varianza acomulada")
            

            graph_6 ,ax= plt.subplots(figsize=(14, 7))
            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel('Número de componentes')
            plt.ylabel('Varianza acumulada')
            plt.grid()
            st.pyplot(graph_6)

            st.header("Análisis de la proporción de relevancias (cargas)")
            st.write("La importancia de cada variable se refleja en la magnitud de los valores en los componentes (mayor magnitud es sinónimo de mayor importancia).")

            st.write("Una forma ad hoc de realizar este análisis es identificando los valores absolutos más altos, con el fin de seleccionar sólo aquellas variables que cumplan con cargas mayores a sólo cierto porcentaje, por ejemplo, sólo realizando la selección de variables con cargas superiores al 50%.")

            st.write("A continuación se muestran los valores absolutos de los componentes principales. Cuanto mayor sea el valor absoluto, más importante es esa variable en el componente principal.")
            pd.DataFrame(abs(pca.components_))

            CargasComponentes = pd.DataFrame(pca.components_, columns=NuevaMatriz.columns)
            st.dataframe(CargasComponentes)



            st.subheader("Selección de características (Elección de variables)")
            st.write("Después de haber utilizado esta herramienta de Análisis de Componentes Princippales, es hora de elegir qué variables deseas excluir del conjunto de datos con el fin de reducir la dimensionalidad y hacer un análisis de datos más eficiente.")
           
            columns_names_actualizado = CargasComponentes.columns.values
            #st.write(columns_names_actualizado)
            st.write()
            variables_eliminadas=st.multiselect("Selecciona las variables que deseas eliminar del análisis del dataframe", columns_names_actualizado, key='Feature_Selection_PCA', help=None, on_change=None, args=None, kwargs=None)
            #st.write(variables_eliminadas)

            #dataframe.index = [""] * len(dataframe)
            #st.table(dataframe)

            #dataframe.assign: st.dataframe(dataframe.assign(hack='').set_index('hack'))
            #st.dataframe(dataframe)
            
            dataframe_PCA_actualizado=CargasComponentes.drop(columns=variables_eliminadas)


            #styler = dataframe.style.hide_index()
            #st.write(styler.to_html(), unsafe_allow_html=True)

            


            st.dataframe(dataframe_PCA_actualizado)


            csv = convert_df(dataframe_PCA_actualizado)

            st.download_button(
               "Haz click aquí para descargar tu nuevo dataframe normalizado y actualziado con las columnas de tu interés",
               csv,
               file_name="DataframeActualizado_PCA.csv",
               

               key='download-csv_pca'
            )




elif componente == 'Clusterización':
    algoritmos3=['Clustering Jerárquico Ascendente', 'Clustering Particional (K-means)']


    menu3= st.sidebar.radio("Realiza la selección del tipo de algoritmo de cluterización que deseas llevar a cabo.", (algoritmos3))


    if menu3 == "Clustering Jerárquico Ascendente":

        st.title("Clustering Jerárquico Ascendente")


        dataset=st.file_uploader("Introduce tu archivo CSV para hacer uso del algoritmo de Clustering Jerárquico Ascendente", type='csv', accept_multiple_files=False, key=5, help="Sólo se permite la carga de un sólo archivo en formato CSV",on_change = None, args = None, kwargs = None)
        
        if dataset is not None:
            st.write("")
            dataframe = pd.read_csv(dataset)
            st.subheader("Visualización del dataframe generado con el archivo cargado")
            st.dataframe(dataframe)

            
            st.subheader("Número de filas y columnas")
            #Obtención del número de filas del dataframe
            filas=dataframe.index
            num_filas=(len(filas))

            #Obtención del número de columnas del dataframe
            colDF=dataframe.columns
            num_columnas=len(colDF)


            col1, col2 = st.columns(2)
            col1.metric('Número de filas del dataset cargado', num_filas)
            col2.metric('Número de columnas del dataset cargado', num_columnas)

            st.header("Breve Análisis Exploratorio y Correlacional de Datos")

            st.subheader("Descripción de la estructura de los datos")
            st.write(explore(dataframe))

            st.subheader("Identificación de datos faltantes")
            st.write("En esta sección podrás ver el número de datos faltantes por cada variable del dataset.")
            st.write(dataframe.isnull().sum())


            st.subheader("Descripción del dataset a través de variables numéricas de clasificación")
            st.write("En esta sección podrás elegir si deseas visualizar cómo es que se distrubuye el dataset cargado de acuerdo a la variable de interés que desees seleccionar.")
            st.write("Recuerda que aunque el menú te permite seleccuinar cualquier variable numérica, sólo se recomienda hacer el análisis con las variables que proporcionen alguna clasificación al conjunto de datos para conocer de una mejor cómo está conformado el dataset.")
            

            #st.write(dataframe.groupby('trabajo').size())


            

            st.subheader("Histograma de la variable de interés")
            

            Num_col_names=dataframe.select_dtypes(include = ["int16", "int32", "int64", "float16", "float32", "float64"]).columns.tolist()
            var_histogram_CJA= st.selectbox("Selecciona la primera variable numérica que tendrá el gráfico de dispersión.", Num_col_names, index=0, key='selectbox_groupby_CJA', help=None, on_change=None, args=None, kwargs=None)
            
            #st.write(var_histogram_CJA)


            if st.button("Presiona este botón en dado caso que requieras de esta funcionalidad", key=3000, help='Si no tienes variables numéricas, entonces no podrás usar esta funcionalidad', on_click=None, args=None, kwargs=None):

                if len(Num_col_names) !=0:    

                    #Conversión de la cadena proveniente del select a un array para poder usarlo como variable en hue 
                        
                    with st.spinner("Esto podría tardar algunos minutos, por favor espera, o en caso contrario, puedes cancelar la creación de la gráfica sin afectar los siguientes apartados."):

                        
                

                        st.subheader('Histograma generado')
                        graph_301 ,ax= plt.subplots(figsize=(8, 4.5))
                        dataframe[var_histogram_CJA].value_counts().plot(kind='bar');
                        st.pyplot(graph_301)

                        st.subheader('Distribución de la variable de interés seleccionada en el conjunto de datos')
                        st.write(dataframe[var_histogram_CJA].value_counts())





                    st.success('Descripción de variable realizada con éxito')

                else:
                    st.warning("El conjunto de datos no tiene variables numéricas, por lo cual no se puede hacer uso de esta función")


            else:

                st.write("")



            st.header('Evaluación Visual')

            #col_names=dataframe.select_dtypes(include='object').columns.tolist()
            st.write("Primero selecciona la variable de interés de la cual te gustaría realizar la evaluación visual, y posteriormente presiona el botón para generarla.")
            variable_seleccionada_CJA_EVAL_VISUAL = st.selectbox("Selecciona la variable de la cual deseas visualizar sus gráficas de dispersión entre pares de variables numéricas.", Num_col_names, index=0, key=302, help=None, on_change=None, args=None, kwargs=None)

            if st.button("Presiona este botón para generar la evaluación visual. Recuerda que puede tardar algunos minutos si tu conjunto de datos es muy grande", key=303, help=None, on_click=None, args=None, kwargs=None):

            

                if len(Num_col_names) !=0:

                    

                    #Conversión de la cadena proveniente del select a un array para poder usarlo como variable en hue 
                        
                    with st.spinner("Esto podría tardar algunos minutos, por favor espera, o en caso contrario, puedes cancelar la creación de la gráfica sin afectar los siguientes apartados."):

                        


                        #graph_4 ,ax= plt.subplots(figsize=(14, 7))
                        x = dataframe
                        fig = sns.pairplot(x, variable_seleccionada_CJA_EVAL_VISUAL)
                        st.pyplot(fig)

                    st.success('Gráfica de dispersión entre pares de variables realizada con éxito')

                else:
                    st.warning("El conjunto de datos no tiene variables numéricas, por lo cual no se puede hacer uso de esta función")



            else:

                st.write("")


            st.header("Gráficas de dispersión entre pares de variables numéricas")

            num_var1 = st.selectbox("Selecciona la primera variable numérica que tendrá el gráfico de dispersión.", Num_col_names, index=0, key=310, help=None, on_change=None, args=None, kwargs=None)
            num_var2 = st.selectbox("Selecciona la segunda variable numérica que tendrá el gráfico de dispersión.", Num_col_names, index=1, key=320, help=None, on_change=None, args=None, kwargs=None)




            graph_303 ,ax= plt.subplots(figsize=(14, 7))
            plt.plot(dataframe[num_var1], dataframe[num_var2], 'o')
            plt.title('Gráfico de dispersión')
            plt.xlabel(num_var1)
            plt.ylabel(num_var2)
            st.pyplot(graph_303)


            st.header("Matriz de correlaciones")

            st.write(dataframe.corr())


            st.subheader("Mapa de calor para el Análisis de la Correlación entre Variables")

            fig305 ,ax= plt.subplots(figsize=(20, 12))
            Matriz = np.triu(dataframe.corr())
            sns.heatmap(dataframe.corr(), cmap='RdBu_r', annot=True, mask=Matriz)
            st.pyplot(fig305)

            st.subheader("Selección de características (Elección de variables)")
            st.write("Después de haber realizado el análisis, selecciona las variables que deseas excluir del conjunto de datos con el fin de pasar a la sección de estandarización de los datos.")
           
            columns_names_CJA = dataframe.columns.values

            variables_eliminadas=st.multiselect("Selecciona las variables que deseas eliminar del análisis del dataframe", columns_names_CJA, key=3100, help=None, on_change=None, args=None, kwargs=None)

            
            dataframe_CJA_actualizado=dataframe.drop(columns=variables_eliminadas)

            st.dataframe(dataframe_CJA_actualizado)




            st.header("Estandarización de los datos")
            normalizar = StandardScaler()                                         # Se instancia el objeto StandardScaler 
            col_names=dataframe_CJA_actualizado.select_dtypes(include='object').columns.tolist()    #Se quitan las variables categórucas para estandarizar sólo variables numéricas
            NuevaMatriz = dataframe_CJA_actualizado.drop(columns=col_names)    # Se quitan las variables no necesarias (nominales)


            clean_dataset(NuevaMatriz)        #limpieza del dataframe antes de estandarizar


            ##Se calcula la media y desviación para cada variable y se normalizan los datos                           
            MNormalizada = normalizar.fit_transform(NuevaMatriz) 
            MNormalizada = pd.DataFrame(MNormalizada, columns=NuevaMatriz.columns)                   

            st.subheader("Matriz Normalizada")

            st.write("A continuación se muestra la matriz con el conjunto de datos normalizado de las variables numericas.")
            st.write(MNormalizada)


            #Obtención del número de filas del dataframe
            filas_normalizadas=MNormalizada.index
            num_filasNorm=(len(filas_normalizadas))

            #Obtención del número de columnas del dataframe
            colDF_Norm=MNormalizada.columns
            num_columnas_Norm=len(colDF_Norm)

            st.write("En esta pequeña sección podrás aprecial el número de filas y columnas de la matriz normalizada.")
            col3, col4 = st.columns(2)
            col3.metric('Número de filas después de la normalización', num_filasNorm)
            col4.metric('Número de columnas después de la normalización', num_columnas_Norm)


            st.header("Clusterización Jerárquica Ascendente")
            st.subheader("Dendograma creado a partir de la clusterización")
            st.write("En esta sección puedes ver de una manera visual los grupos de datos que se formaron al implementar el algoritmo")

            metodosCJA = ['single','complete', 'average', 'weighted', 'centroid' ]
            distanciasCJA = ['euclidean', 'cityblock','minkowski', 'cityblock','sqeuclidean', 'cosine', 'chebyshev','canberra', 'braycurtis']


            metodo_CJA = st.selectbox("Selecciona el método con el cual quieres generar el dendograma", metodosCJA, index=1, key=3601, help=None, on_change=None, args=None, kwargs=None)
            metrica_CJA = st.selectbox("Selecciona el tipo de métrica de distancia con la cual quieres generar el dendograma", distanciasCJA, index=5 ,key=3602, help=None, on_change=None, args=None, kwargs=None)
            #st.write(metodo_CJA)

            #https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
            #https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist

    
            with st.spinner("Esto podría tardar algunos segundos, por favor espera."):

                
        

                graph_305 ,ax= plt.subplots(figsize=(14, 7))
                plt.title("Dendograma generado con Clustering Ascendente Jerárquico")
                plt.xlabel('Datos')
                plt.ylabel('Distancia')
                Arbol = shc.dendrogram(shc.linkage(MNormalizada, method=metodo_CJA, metric=metrica_CJA))
                st.pyplot(graph_305)
                

            st.success('Dendograma creado con éxito')



            st.subheader("Creación de las etiquetas")

            num_clusters = st.number_input('Introduce el número de clústers (grupos) que obtuviste después de la configuración anterior con la cual generaste el clúster', min_value=2, max_value=None, step=1, format=None, key='num_clusters_CJA', help='Sólo se admiten valores numéricos', on_change=None, args=None, kwargs=None)
            num_clusters= int(num_clusters)
            MJerarquico = AgglomerativeClustering(n_clusters=num_clusters, linkage='complete', affinity='euclidean')
            MJerarquico.fit_predict(MNormalizada)

            #st.write("Tabla de datos con etiquetas")
            #st.write(MJerarquico.labels_)

            st.subheader("Dataframe actualizado con datos etiquetados de acuerdo al número de clúster al que pertenecen")
            df_etiquetado_CJA=dataframe_CJA_actualizado
            df_etiquetado_CJA['Cluster'] = MJerarquico.labels_
            st.dataframe(df_etiquetado_CJA)




            st.subheader("Cantidad de elementos en los clústers")

            graph_307 ,ax= plt.subplots(figsize=(8, 4.5))
            df_etiquetado_CJA['Cluster'].value_counts().plot(kind='bar');
            st.pyplot(graph_307)

            st.write("La siguiente tabla muestra el número de elementos exactos que contiene cada uno de los clústeres generados")
            st.write(df_etiquetado_CJA.groupby(['Cluster'])['Cluster'].count())


            st.subheader("Visualización de los datos de un clúster específico")


            lst_num_clusters = list(range(0,num_clusters))
            arr_num_clusters=np.array(lst_num_clusters)
            #lista_clusters=range(num_clusters)
            #st.write(arr_num_clusters)

            select_cluster_CJA = st.selectbox("Selecciona el número de clúster del cual quieres ver su información", arr_num_clusters, index=0, key=3610, help=None, on_change=None, args=None, kwargs=None)
            
            st.write(df_etiquetado_CJA[df_etiquetado_CJA.Cluster == select_cluster_CJA])

            st.subheader("Análisis de estadísticas promedio por clúster generado")
            st.write("A través de la siguiente tabla, se presentan las estadísticas promedio de cada variable numérica respecto a cada cluster generado con el algoritmo y los respectivos parámetros seleccionados anteriormente.")

            CentroidesH = df_etiquetado_CJA.groupby('Cluster').mean()
            st.dataframe(CentroidesH)


    elif menu3 == 'Clustering Particional (K-means)':

        st.title("Clustering Particional (K-means)")







        dataset=st.file_uploader("Introduce tu archivo CSV para hacer uso del algoritmo de Clustering Particional", type='csv', accept_multiple_files=False, key=6, help="Sólo se permite la carga de un sólo archivo en formato CSV",on_change = None, args = None, kwargs = None)
        
        if dataset is not None:
            st.write("")
            dataframe = pd.read_csv(dataset)
            st.subheader("Visualización del dataframe generado con el archivo cargado")
            st.dataframe(dataframe)

            
            st.subheader("Número de filas y columnas")
            #Obtención del número de filas del dataframe
            filas=dataframe.index
            num_filas=(len(filas))

            #Obtención del número de columnas del dataframe
            colDF=dataframe.columns
            num_columnas=len(colDF)


            col1, col2 = st.columns(2)
            col1.metric('Número de filas del dataset cargado', num_filas)
            col2.metric('Número de columnas del dataset cargado', num_columnas)

            st.header("Breve Análisis Exploratorio y Correlacional de Datos")

            st.subheader("Descripción de la estructura de los datos")
            st.write(explore(dataframe))

            st.subheader("Identificación de datos faltantes")
            st.write("En esta sección podrás ver el número de datos faltantes por cada variable del dataset.")
            st.write(dataframe.isnull().sum())


            st.subheader("Descripción del dataset a través de variables numéricas de clasificación")
            st.write("En esta sección podrás elegir si deseas visualizar cómo es que se distrubuye el dataset cargado de acuerdo a la variable de interés que desees seleccionar.")
            st.write("Recuerda que aunque el menú te permite seleccuinar cualquier variable numérica, sólo se recomienda hacer el análisis con las variables que proporcionen alguna clasificación al conjunto de datos para conocer de una mejor cómo está conformado el dataset.")
            



            

            st.subheader("Histograma de la variable de interés")
            

            Num_col_names=dataframe.select_dtypes(include = ["int16", "int32", "int64", "float16", "float32", "float64"]).columns.tolist()
            var_histogram_CP= st.selectbox("Selecciona la primera variable numérica que tendrá el gráfico de dispersión.", Num_col_names, index=0, key='selectbox_groupby_CPart', help=None, on_change=None, args=None, kwargs=None)
            



            if st.button("Presiona este botón en dado caso que requieras de esta funcionalidad", key=4000, help='Si no tienes variables numéricas, entonces no podrás usar esta funcionalidad', on_click=None, args=None, kwargs=None):

                if len(Num_col_names) !=0:    

                    #Conversión de la cadena proveniente del select a un array para poder usarlo como variable en hue 
                        
                    with st.spinner("Esto podría tardar algunos minutos, por favor espera, o en caso contrario, puedes cancelar la creación de la gráfica sin afectar los siguientes apartados."):

                        
                

                        st.subheader('Histograma generado')
                        graph_401 ,ax= plt.subplots(figsize=(8, 4.5))
                        dataframe[var_histogram_CP].value_counts().plot(kind='bar');
                        st.pyplot(graph_401)

                        st.subheader('Distribución de la variable de interés seleccionada en el conjunto de datos')
                        st.write(dataframe[var_histogram_CP].value_counts())





                    st.success('Descripción de variable realizada con éxito')

                else:
                    st.warning("El conjunto de datos no tiene variables numéricas, por lo cual no se puede hacer uso de esta función")


            else:

                st.write("")



            st.header('Evaluación Visual')

            #col_names=dataframe.select_dtypes(include='object').columns.tolist()
            st.write("Primero selecciona la variable de interés de la cual te gustaría realizar la evaluación visual, y posteriormente presiona el botón para generarla.")
            variable_seleccionada_CP_EVAL_VISUAL = st.selectbox("Selecciona la variable de la cual deseas visualizar sus gráficas de dispersión entre pares de variables numéricas.", Num_col_names, index=0, key=402, help=None, on_change=None, args=None, kwargs=None)

            if st.button("Presiona este botón para generar la evaluación visual. Recuerda que puede tardar algunos minutos si tu conjunto de datos es muy grande", key=403, help=None, on_click=None, args=None, kwargs=None):

            

                if len(Num_col_names) !=0:

                    

                    #Conversión de la cadena proveniente del select a un array para poder usarlo como variable en hue 
                        
                    with st.spinner("Esto podría tardar algunos minutos, por favor espera, o en caso contrario, puedes cancelar la creación de la gráfica sin afectar los siguientes apartados."):

                        


                        #graph_4 ,ax= plt.subplots(figsize=(14, 7))
                        x = dataframe
                        fig = sns.pairplot(x, variable_seleccionada_CP_EVAL_VISUAL)
                        st.pyplot(fig)

                    st.success('Gráfica de dispersión entre pares de variables realizada con éxito')

                else:
                    st.warning("El conjunto de datos no tiene variables numéricas, por lo cual no se puede hacer uso de esta función")



            else:

                st.write("")


            st.header("Gráficas de dispersión entre pares de variables numéricas")

            num_var1 = st.selectbox("Selecciona la primera variable numérica que tendrá el gráfico de dispersión.", Num_col_names, index=0, key=410, help=None, on_change=None, args=None, kwargs=None)
            num_var2 = st.selectbox("Selecciona la segunda variable numérica que tendrá el gráfico de dispersión.", Num_col_names, index=1, key=420, help=None, on_change=None, args=None, kwargs=None)




            graph_403 ,ax= plt.subplots(figsize=(14, 7))
            plt.plot(dataframe[num_var1], dataframe[num_var2], 'o')
            plt.title('Gráfico de dispersión')
            plt.xlabel(num_var1)
            plt.ylabel(num_var2)
            st.pyplot(graph_403)


            st.header("Matriz de correlaciones")

            st.write(dataframe.corr())


            st.subheader("Mapa de calor para el Análisis de la Correlación entre Variables")

            fig305 ,ax= plt.subplots(figsize=(20, 12))
            Matriz = np.triu(dataframe.corr())
            sns.heatmap(dataframe.corr(), cmap='RdBu_r', annot=True, mask=Matriz)
            st.pyplot(fig305)

            st.subheader("Selección de características (Elección de variables)")
            st.write("Después de haber realizado el análisis, selecciona las variables que deseas excluir del conjunto de datos con el fin de pasar a la sección de estandarización de los datos.")
           
            columns_names_CP = dataframe.columns.values

            variables_eliminadas=st.multiselect("Selecciona las variables que deseas eliminar del análisis del dataframe", columns_names_CP, key=4100, help=None, on_change=None, args=None, kwargs=None)

            
            dataframe_CP_actualizado=dataframe.drop(columns=variables_eliminadas)

            st.dataframe(dataframe_CP_actualizado)



            st.header("Estandarización de los datos")
            normalizar = StandardScaler()                                         # Se instancia el objeto StandardScaler 
            col_names=dataframe_CP_actualizado.select_dtypes(include='object').columns.tolist()    #Se quitan las variables categórucas para estandarizar sólo variables numéricas
            NuevaMatriz = dataframe_CP_actualizado.drop(columns=col_names)    # Se quitan las variables no necesarias (nominales)


            clean_dataset(NuevaMatriz)        #limpieza del dataframe antes de estandarizar


            ##Se calcula la media y desviación para cada variable y se normalizan los datos                           
            MNormalizada = normalizar.fit_transform(NuevaMatriz) 
            MNormalizada = pd.DataFrame(MNormalizada, columns=NuevaMatriz.columns)                   

            st.subheader("Matriz Normalizada")

            st.write("A continuación se muestra la matriz con el conjunto de datos normalizado de las variables numericas.")
            st.write(MNormalizada)


            #Obtención del número de filas del dataframe
            filas_normalizadas=MNormalizada.index
            num_filasNorm=(len(filas_normalizadas))

            #Obtención del número de columnas del dataframe
            colDF_Norm=MNormalizada.columns
            num_columnas_Norm=len(colDF_Norm)

            st.write("En esta pequeña sección podrás aprecial el número de filas y columnas de la matriz normalizada.")
            col3, col4 = st.columns(2)
            col3.metric('Número de filas después de la normalización', num_filasNorm)
            col4.metric('Número de columnas después de la normalización', num_columnas_Norm)



            st.header("Clusterización Particional")
            st.subheader("Método del codo")
            st.write("A continuación, se presenta la gráfica del método del codo como herramienta de apoyo en la decisión del número adecuado de clústers que se desean obtener con este algoritmo de clusterización particional.")

            st.write("La idea básica de los algoritmos de clustering es la minimización de la varianza intra-cluster y la maximización de la varianza inter-cluster. Es decir, queremos que cada observación se encuentre muy cerca a las de su mismo grupo y los grupos lo más lejos posible entre ellos.")

            st.write("El método del codo utiliza la distancia media de las observaciones a su centroide. Es decir, se fija en las distancias intra-cluster. Cuanto más grande es el número de clusters k, la varianza intra-cluster tiende a disminuir")

            #https://machinelearningparatodos.com/segmentacion-utilizando-k-means-en-python/
            


            st.write("Cabe destacar que con la ayuda de esta gráfica, se puede detectar a través de un cambio en la dirección de la gráfica el punto óptimo que nos indica el valor adecuado del número de clústers, por lo cual se incluye su funcionalidad en la plataforma.")

            #Definición de k clusters para K-means
            #Se utiliza random_state para inicializar el generador interno de números aleatorios
            
            SSE = []
            for i in range(2, 12):
                km = KMeans(n_clusters=i, random_state=0)
                km.fit(MNormalizada)
                SSE.append(km.inertia_)

            #Se grafica SSE en función de k
            graph_501 ,ax= plt.subplots(figsize=(14, 7))
            plt.plot(range(2, 12), SSE, marker='o')
            plt.xlabel('Cantidad de clusters *k*')
            plt.ylabel('SSE')
            plt.title('Elbow Method')
            st.pyplot(graph_501)  


            st.subheader("Método del codo con detección automática de valor de número de clusters óptimo") 

            st.write("A través de la siguiente gráfica generada con la biblioteca Kneed, podrás decidir de una manera más precisa el número de clústers óptimo")


            #SECCIÓN EN DOS COLUMNAS
            col7,col8=st.columns(2)

            curves=['concave', 'convex']
            directions = ['increasing','decreasing']

            with col7:    #Columna 1 para selección del tipo de curva
                curve_CP = st.radio("Selecciona el tipo de curva con la que deseas generar la gráfica:", (curves), index=1)
                

            with col8:    #Columna 2 para selección del tipo de dirección
                direction_CP =st.radio("Selecciona el tipo de dirección con la que deseas generar la gráfica:", (directions), index=1)
                
                 
            st.write("A continuación se muestra el número de clusters recomendado")


            kl = KneeLocator(range(2, 12), SSE, curve=curve_CP, direction=direction_CP)
            
            st.subheader(kl.elbow)

            st.write("Gráfica generada:")


            #https://kneed.readthedocs.io/en/stable/parameters.html#curve

            graph_502 ,ax= plt.subplots(figsize=(14, 7))
            

            plt.style.use('ggplot')
            
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(kl.plot_knee())  






            st.subheader("Creación de las etiquetas")

            num_clusters = st.number_input('Introduce el número de clústers (grupos) que obtuviste después de la configuración anterior con la cual generaste el clúster', min_value=2, max_value=None, step=1, format=None, key='num_clusters_CP', help='Sólo se admiten valores numéricos', on_change=None, args=None, kwargs=None)
            num_clusters= int(num_clusters)
            

            MParticional = KMeans(n_clusters=num_clusters, random_state=0).fit(MNormalizada)
            MParticional.predict(MNormalizada)
            #st.write(MParticional.labels_)


            #st.write("Tabla de datos con etiquetas")
            #st.write(MJerarquico.labels_)

            st.subheader("Dataframe actualizado con datos etiquetados de acuerdo al número de clúster al que pertenecen")
            df_etiquetado_CP=dataframe_CP_actualizado
            df_etiquetado_CP['Cluster'] = MParticional.labels_
            st.dataframe(df_etiquetado_CP)




            st.subheader("Cantidad de elementos en los clústers")

            graph_507 ,ax= plt.subplots(figsize=(8, 4.5))
            df_etiquetado_CP['Cluster'].value_counts().plot(kind='bar');
            st.pyplot(graph_507)

            st.write("La siguiente tabla muestra el número de elementos exactos que contiene cada uno de los clústeres generados")
            st.write(df_etiquetado_CP.groupby(['Cluster'])['Cluster'].count())


            st.subheader("Visualización de los datos de un clúster específico")


            lst_num_clusters = list(range(0,num_clusters))
            arr_num_clusters=np.array(lst_num_clusters)
            #lista_clusters=range(num_clusters)
            #st.write(arr_num_clusters)

            select_cluster_CP = st.selectbox("Selecciona el número de clúster del cual quieres ver su información", arr_num_clusters, index=0, key=3610, help=None, on_change=None, args=None, kwargs=None)
            
            st.write(df_etiquetado_CP[df_etiquetado_CP.Cluster == select_cluster_CP])

            st.subheader("Análisis de estadísticas promedio por clúster generado")
            st.write("A través de la siguiente tabla, se presentan las estadísticas promedio de cada variable numérica respecto a cada cluster generado con el algoritmo y los respectivos parámetros seleccionados anteriormente.")

            CentroidesH = df_etiquetado_CP.groupby('Cluster').mean()
            st.dataframe(CentroidesH)



elif componente == 'Reglas de Asociación':
    algoritmos4=['Algoritmo A Priori']


    menu4= st.sidebar.radio("Este componente sólo cuenta con el algoritmo A Priori.", (algoritmos4))


    if menu4 == "Algoritmo A Priori":

        st.title("Algoritmo A Priori")


        dataset=st.file_uploader("Introduce tu archivo CSV para hacer uso del algoritmo A Priori", type='csv', accept_multiple_files=False, key=7, help="Sólo se permite la carga de un sólo archivo en formato CSV",on_change = None, args = None, kwargs = None)
        
        if dataset is not None:
            st.write("")
            
            st.subheader("Visualización del dataframe generado con el archivo cargado")
            

            df_options=['El dataframe cargado tiene headers','El dataframe cargado no tiene headers']
            st.write("Para una mejor lectura del conjuto de datos, indíca si tu dataframe tiene headers o no a través de la siguiente sección.")
            df_option_selected = st.radio("Selecciona por favor la configuración que tiene el dataframe", (df_options), index=0)
                
            if df_option_selected == 'El dataframe cargado tiene headers':
                dataframe = pd.read_csv(dataset)

            elif df_option_selected == 'El dataframe cargado no tiene headers':
                dataframe = pd.read_csv(dataset, header=None)
                

            st.dataframe(dataframe)

            st.subheader("Número de filas y columnas")
            #Obtención del número de filas del dataframe
            filas=dataframe.index
            num_filas=(len(filas))

            #Obtención del número de columnas del dataframe
            colDF=dataframe.columns
            num_columnas=len(colDF)


            col1, col2 = st.columns(2)
            col1.metric('Número de filas del dataset cargado', num_filas)
            col2.metric('Número de columnas del dataset cargado', num_columnas)


            st.header("Exploración del conjunto de datos")
            st.write("Antes de ejecutar el algoritmo es ampliamente recomendable observar la distribución de la frecuencia de los elementos.")
            st.write('Para ello, se generará una gráfica de frecuencia para visualizar los items para importantes del dataset, por lo cual se procede a generar una tabla de frecuencia respecto a cada item.')
            

            Transacciones = dataframe.values.reshape(-1).tolist() #-1 significa 'dimensión no conocida'
            
            apartado_trasacciones_df=Transacciones

            #Se crea una matriz (dataframe) usando la lista y se incluye una columna 'Frecuencia'
            Lista = pd.DataFrame(Transacciones)
            Lista['Frecuencia'] = 1

            #Se agrupa los elementos
            Lista = Lista.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=False) #Conteo
            Lista['Porcentaje'] = (Lista['Frecuencia'] / Lista['Frecuencia'].sum()) #Porcentaje
            Lista = Lista.rename(columns={0 : 'Item'})

            #Se muestra la lista
            st.dataframe(Lista)


            if st.button("Si deseas visualizar la lista de transacciones, presiona este botón.", key=601, help='Esta sección puede ser de utilidad si deseas conocer el número total de transacciones', on_click=None, args=None, kwargs=None):
                st.dataframe(apartado_trasacciones_df)    #visualización de lista de transacciones generada


                filas_lista_transacciones=apartado_trasacciones_df.index
                num_filas_transacciones=(len(apartado_trasacciones_df))
                st.write("Número de transacciones:")
                st.subheader(num_filas_transacciones)


            else:

                st.write("")


            st.header("Gráfica de barras de frecuencia por cada item del conjunto de datos")

            graph_601 ,ax= plt.subplots(figsize=(16, 20), dpi=1200)
            plt.ylabel('Item')
            plt.xlabel('Frecuencia')
            plt.barh(Lista['Item'], width=Lista['Frecuencia'], color='blue')

            st.pyplot(graph_601)


            st.header("Algoritmo A Priori")
            st.subheader("Preparación de la data")
            st.write("La función Apriori de Python requiere que el conjunto de datos tenga la forma de una lista de listas, donde cada transacción es una lista interna dentro de una gran lista. Los datos actuales están en un dataframe de Pandas, por lo que, se requiere convertir en una lista.")


            TransaccionesLista = dataframe.stack().groupby(level=0).apply(list).tolist()


            NuevaLista = []

            for item in TransaccionesLista:
              if str(item) != 'nan':
                NuevaLista.append(item)
            #print(NuevaLista)
            #st.write(NuevaLista)



            if st.button("Si deseas visualizar la lista de listas generada en esta preparación de la data, presiona este botón.", key=602, help='Esta sección puede ser de utilidad si deseas conocer el número total de transacciones', on_click=None, args=None, kwargs=None):
                st.write(NuevaLista)


            else:
                st.write("")




            #Visualización de una lista en específico
            st.subheader("Visualización de una lista en específico")

            num_lista_RA = st.number_input('Introduce el número de la lista N que deseas visualizar', min_value=0, max_value=num_filas-1, step=1, format=None, key='num_lista_apartadoextraRA', help='Sólo se admiten valores numéricos enteros positivos', on_change=None, args=None, kwargs=None)
            st.write("El máximo número que puedes ingresar es:", num_filas-1)
            num_lista_RA= int(num_lista_RA)    


            st.write(NuevaLista[num_lista_RA])    



   




            st.header("Aplicación del algoritmo")
            st.write("En esta sección, deberás de introducir los valores requeridos para llevar a cabo la generación de reglas de asociación mediante el algoritmo A priori, los cuales son el soporte, la confianza y la elevación.")

            st.write("Recuerda que el soporte mínimo definirá el grado de importancia que tendrá la regla. Como recomnendación y a manera de ejemplo, pueden calcularse a través del cálculo de aquellos productos que se hayan transaccionado al menos n veces por día o semana respecto a las operaciones llevadas a cabo por cada cliente")

            st.write("La confianza mínima es la fiabilidad de la regla, la cual de acuerdo al valor n establecido establecerá que el n porciento de las transacciones tendrán esa característica.")

            st.write("Finalmente, deberás seleccionar también el grado de elevación que deseas tengan las reglas de asociación. Este valor n hará que la probabilidad de transaccionar ese item aumente n veces, y siempre debe ser mayor a uno.")





            st.subheader("Parámetros de generación de reglas de asociación")
            st.write("Una vez aclarado lo anteror, introduce los parámetros con los cuales quieres ejecutar el algoritmo para generar las reglas.")
            
            valor_soporte_min = st.number_input('Introduce el valor del soporte mínimo que deseas utilizar', min_value=0.0001, max_value=1.0, format="%.4f", key='valor_RA_SOPORTE', help='Sólo se admiten valores numéricos positivos entre 0.0001 y 1', on_change=None, args=None, kwargs=None)
            #st.write(valor_soporte)


            valor_confianza = st.number_input('Introduce el valor del porcentaje que deses utilizar para el parámetro de la confianza', min_value=0.0001, max_value=1.0, format="%.4f", key='valor_RA_confianza', help='Sólo se admiten valores numéricos positivos entre 0.0001 y 1', on_change=None, args=None, kwargs=None)
            #st.write(valor_confianza)

            valor_elevacion = st.number_input('Introduce el valor de elevación que deseas utilizar. Recuerda que debe ser mayor a uno', min_value=1.001, max_value=None, format="%.3f", key='valor_RA_elevacion', help='Sólo se admiten valores numéricos mayores a 1.001', on_change=None, args=None, kwargs=None)
            #st.write(valor_soporte)

            ReglasC1 = apriori(TransaccionesLista, 
                   min_support = valor_soporte_min, 
                   min_confidence = valor_confianza, 
                   min_lif = valor_elevacion)





            if st.button("Presiona este botón para generar las reglas de asociación", key=605, help=None, on_click=None, args=None, kwargs=None):

                        
                with st.spinner("Esto podría tardar algunos segundos, por favor espera."):


                    ResultadosC1 = list(ReglasC1)

                    for item in ResultadosC1:

                        #El primer índice de la lista
                        Emparejar = item[0]
                        items = [x for x in Emparejar]
                        st.write("Regla: " + str(item[0]))

                        #El segundo índice de la lista
                        st.write("Soporte: " + str(item[1]))

                        #El tercer índice de la lista
                        st.write("Confianza: " + str(item[2][0][2]))
                        st.write("Li: " + str(item[2][0][3])) 
                        st.write("=====================================") 
                            
            

                st.success('Reglas de asociación generadas exitosamente')


            else:

                st.write("")







if componente == 'Árboles de decisión':
    algoritmos5=['Árbol De Decisión (Pronóstico)', 'Árbol De Decisión (Clasificación)']


    menu5 = st.sidebar.radio("Realiza la selección del tipo de Árbol de Decisión que deseas usar", (algoritmos5))

    if menu5 == "Árbol De Decisión (Pronóstico)":

        st.title("Árbol De Decisión (Pronóstico)")



        dataset=st.file_uploader("Introduce tu archivo CSV para hacer uso del algoritmo de Clustering Particional", type='csv', accept_multiple_files=False, key=8, help="Sólo se permite la carga de un sólo archivo en formato CSV",on_change = None, args = None, kwargs = None)
        
        if dataset is not None:
            st.write("")
            dataframe = pd.read_csv(dataset)
            st.subheader("Visualización del dataframe generado con el archivo cargado")
            st.dataframe(dataframe)


            st.subheader("Número de filas y columnas")
            #Obtención del número de filas del dataframe
            filas=dataframe.index
            num_filas=(len(filas))

            #Obtención del número de columnas del dataframe
            colDF=dataframe.columns
            num_columnas=len(colDF)


            col1, col2 = st.columns(2)
            col1.metric('Número de filas del dataset cargado', num_filas)
            col2.metric('Número de columnas del dataset cargado', num_columnas)






            st.subheader("Resumen Estadístico de Variables Numéricas")
            st.caption("En este apartado podrás apreciar las estadísticas más reperesentativas de cada variable numérica, tales como el recuento del número de registros, el promedio, la desviación estandar, el mínimo y máximo, así como también la información de los cuartiles del dataset.")
            st.write(dataframe.describe())






            st.header("Matriz de correlaciones")

            st.write(dataframe.corr())


            st.subheader("Mapa de calor para el Análisis de la Correlación entre Variables")

            fig801 ,ax= plt.subplots(figsize=(20, 12))
            Matriz = np.triu(dataframe.corr())
            sns.heatmap(dataframe.corr(), cmap='RdBu_r', annot=True, mask=Matriz)
            st.pyplot(fig801)

            st.subheader("Selección de características (Elección de variables)")
            st.write("Después de haber visualizado el heatmap, selecciona las variables que deseas excluir del conjunto de datos (si es que deseas omitir alguna).")
           
            columns_names_AD_pronostico = dataframe.columns.values

            variables_eliminadas=st.multiselect("Selecciona las variables que deseas eliminar del análisis del dataframe",  columns_names_AD_pronostico, key=8100, help=None, on_change=None, args=None, kwargs=None)

            
            dataframe_AD_Pronostico=dataframe.drop(columns=variables_eliminadas)

            st.dataframe(dataframe_AD_Pronostico)


            st.header("Aplicación del algoritmo de Árbol de Decisión para Pronóstico")
            st.subheader("Selección de las variables predictoras (X)")


            columns_names_AD_pronostico_actualizado=dataframe_AD_Pronostico.columns.values

            variables_seleccionadas_ADP_x=st.multiselect("Selecciona las variables predictoras", columns_names_AD_pronostico_actualizado, default=dataframe_AD_Pronostico.columns.values[0], key=8101, help=None, on_change=None, args=None, kwargs=None)



            #st.write(variables_seleccionadas_ADP_x)


            select_var_pronostico = st.selectbox("Selecciona la variable a pronosticar", columns_names_AD_pronostico_actualizado, index=0, key=8102, help=None, on_change=None, args=None, kwargs=None)
            
            #st.write(select_var_pronostico)

            X = np.array(dataframe_AD_Pronostico[variables_seleccionadas_ADP_x])

            #st.write(variables_)
            
            st.subheader("Dataframe de las variables predictoras")
            st.dataframe(X)


            st.subheader("Dataframe de las variable a pronosticar")
            Y = np.array(dataframe_AD_Pronostico[select_var_pronostico])
            st.dataframe(Y)



            st.subheader("División del conjunto de datos")

            porcentaje_validacion = st.number_input('Selecciona por favor qué porcentaje del dataset utilizarás para la validación del modelo', min_value=0.01, max_value=0.40, format="%.2f", key='valor_validacion_adp', help='Sólo se admiten valores numéricos positivos entre 0.01 y 0.40', on_change=None, args=None, kwargs=None)
            #st.write(porcentaje_validacion)


            if st.button("Presiona este botón para generar la división del dataset y para contiinuar con el algoritmo", key=815, help=None, on_click=None, args=None, kwargs=None):
                X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = porcentaje_validacion, random_state = 0, shuffle = True)

                st.subheader("Visualización del dataframe de entrenamiento de variables predictoras")
                st.dataframe(X_train)


            
                st.subheader("Visualización del dataframe de entrenamiento de la variable a predecir")
                st.dataframe(Y_train)


                st.subheader("Entrenamiento del modelo (prónóstico generado)")
                st.write("En esta sección podrás apreciar el pronóstico generado después de entrenar al modelo. Los resultados provienen del porcentaje del conjunto de datos de validación seleccionado previamente.")

                PronosticoAD = DecisionTreeRegressor()
                PronosticoAD.fit(X_train, Y_train)
                #Se genera el pronóstico
                Y_Pronostico = PronosticoAD.predict(X_test)
                st.dataframe(Y_Pronostico)


                st.subheader("Comparación entre los valores reales de los datos de validación vs los valores predecidos con el modelo")
                st.write("El siguiente dataframe muestra en la primera columna los valores reales del conjunto de datos de validación y en la segunda columna se muestran los valores predecidos con el modelo.")
                Valores = pd.DataFrame(Y_test, Y_Pronostico)
                st.dataframe(Valores) 


                st.write("Gráfica comparativa entre los valores reales y los valores predecidos")



                graph_810 ,ax= plt.subplots(figsize=(20,10))

                plt.plot(Y_test, color='green', marker='o', label='Y_test')
                plt.plot(Y_Pronostico, color='red', marker='o', label='Y_Pronostico')
                plt.xlabel('Número de dato')
                plt.ylabel(select_var_pronostico)
                plt.title('Valores reales vs Valores predecidos')
                plt.grid(True)
                plt.legend()

                st.pyplot(graph_810)


                st.subheader("Porcentaje de precisión del modelo")
                score_model=r2_score(Y_test, Y_Pronostico)
                st.write(score_model)


                st.subheader("Obtención de los parámetros del modelo")
                st.write('Criterio: \n', PronosticoAD.criterion)
                st.write("MAE: %.4f" % mean_absolute_error(Y_test, Y_Pronostico))
                st.write("MSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico))
                st.write("RMSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico, squared=False))   #True devuelve MSE, False devuelve RMSE
                st.write('Score: %.4f' % r2_score(Y_test, Y_Pronostico))


                st.subheader("Importancia de cada variable predictora del modelo")


                Importancia = pd.DataFrame({'Variable': list(dataframe_AD_Pronostico[variables_seleccionadas_ADP_x]),
                            'Importancia': PronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)
                
                st.write(Importancia)


                st.subheader("Visualización del árbol")
                



                graph_811 ,ax= plt.subplots(figsize=(20,20))
                
                
                plot_tree(PronosticoAD, feature_names = list(dataframe_AD_Pronostico[variables_seleccionadas_ADP_x]))


                st.pyplot(graph_811)


                st.subheader("Nuevos Pronósticos")
                #st.write(list(dataframe_AD_Pronostico[variables_seleccionadas_ADP_x]))
                
                #listaSSP=list(dataframe_AD_Pronostico[variables_seleccionadas_ADP_x])
                #num_variables_seleccionadas=len(listaSSP)
                #st.write(num_variables_seleccionadas)
                
                #arraySSP= np.array(listaSSP)
                #st.write(arraySSP)


                #Intento de implementación de Sistema Automatizado de Nuevos Pronósticos

                #for i in range(1,num_variables_seleccionadas): #add n rows of data

                    #dataframe_AD_Pronostico.loc[i, [dataframe_AD_Pronostico[i]]] = 0.1
                    #df.loc[i, ['B']] = randint(0,99)
                    #df.loc[i, ['C']] = randint(0,99)
                    #df.loc[i, ['D']] = randint(0,99)
                    #df.loc[i, ['E']] = randint(0,99)
                #st.dataframe(dataframe_AD_Pronostico)

                valor_textura = st.number_input('Introduce el valor de Texture', key='valor_textura',value=10.38, format="%.4f")
  

                valor_perimetro = st.number_input('Introduce el valor de Perimeter', key='valor_perimeter',value=120.8, format="%.4f")


                valor_Smoothness = st.number_input('Introduce el valor de Smoothness', key='valor_Smoothness',value=0.11840, format="%.4f")
                
                valor_Compactness = st.number_input('Introduce el valor de Compactness', key='valor_Compactness',value=0.27760, format="%.4f")


                valor_Symmetry = st.number_input('Introduce el valor de Symmetry', key='valor_Symmetry',value=0.2419, format="%.4f")

                valor_FractalDimension = st.number_input('Introduce el valor de FractalDimension', key='valor_FractalDimension',value=0.07871, format="%.4f")





                AreaTumorID1 = pd.DataFrame({'Texture': [valor_textura], 
                                             'Perimeter': [valor_perimetro], 
                                             'Smoothness': [valor_Smoothness], 
                                             'Compactness': [valor_Compactness], 
                                             'Symmetry': [valor_Symmetry], 
                                             'FractalDimension': [valor_FractalDimension]})

                st.subheader("Datos recopilados de las variables predictoras")
                st.dataframe(AreaTumorID1)

                st.subheader("Pronóstico generado con el modelo")
                df_pronostico=PronosticoAD.predict(AreaTumorID1)
                st.write('El pronóstico es de:')
                st.info(df_pronostico[0])


            else:
                st.write("")



        





    elif menu5 == 'Árbol De Decisión (Clasificación)':


        st.title('Árbol De Decisión (Clasificación)')





























        dataset=st.file_uploader("Introduce tu archivo CSV para hacer uso del algoritmo de Árbol De Decisión (Clasificación)", type='csv', accept_multiple_files=False, key=9, help="Sólo se permite la carga de un sólo archivo en formato CSV",on_change = None, args = None, kwargs = None)
        
        if dataset is not None:
            st.write("")
            dataframe = pd.read_csv(dataset)
            st.subheader("Visualización del dataframe generado con el archivo cargado")
            st.dataframe(dataframe)


            st.subheader("Número de filas y columnas")
            #Obtención del número de filas del dataframe
            filas=dataframe.index
            num_filas=(len(filas))

            #Obtención del número de columnas del dataframe
            colDF=dataframe.columns
            num_columnas=len(colDF)


            col1, col2 = st.columns(2)
            col1.metric('Número de filas del dataset cargado', num_filas)
            col2.metric('Número de columnas del dataset cargado', num_columnas)






            st.subheader("Resumen Estadístico de Variables Numéricas")
            st.caption("En este apartado podrás apreciar las estadísticas más reperesentativas de cada variable numérica, tales como el recuento del número de registros, el promedio, la desviación estandar, el mínimo y máximo, así como también la información de los cuartiles del dataset.")
            st.write(dataframe.describe())






            st.header("Matriz de correlaciones")

            st.write(dataframe.corr())


            st.subheader("Mapa de calor para el Análisis de la Correlación entre Variables")

            fig801 ,ax= plt.subplots(figsize=(20, 12))
            Matriz = np.triu(dataframe.corr())
            sns.heatmap(dataframe.corr(), cmap='RdBu_r', annot=True, mask=Matriz)
            st.pyplot(fig801)

            st.subheader("Selección de características (Elección de variables)")
            st.write("Después de haber visualizado el heatmap, selecciona las variables que deseas excluir del conjunto de datos (si es que deseas omitir alguna).")
           
            columns_names_AD_pronostico = dataframe.columns.values

            variables_eliminadas=st.multiselect("Selecciona las variables que deseas eliminar del análisis del dataframe",  columns_names_AD_pronostico, key=9100, help=None, on_change=None, args=None, kwargs=None)

            
            dataframe_AD_Pronostico=dataframe.drop(columns=variables_eliminadas)

            st.dataframe(dataframe_AD_Pronostico)


            st.header("Aplicación del algoritmo de Árbol de Decisión para Pronóstico")
            st.subheader("Selección de las variables predictoras (X)")


            columns_names_AD_pronostico_actualizado=dataframe_AD_Pronostico.columns.values

            variables_seleccionadas_ADP_x=st.multiselect("Selecciona las variables predictoras", columns_names_AD_pronostico_actualizado, default=dataframe_AD_Pronostico.columns.values[0], key=9101, help=None, on_change=None, args=None, kwargs=None)



            #st.write(variables_seleccionadas_ADP_x)


            select_var_pronostico = st.selectbox("Selecciona la variable clase", columns_names_AD_pronostico_actualizado, index=0, key=9102, help=None, on_change=None, args=None, kwargs=None)
            
            #st.write(select_var_pronostico)

            X = np.array(dataframe_AD_Pronostico[variables_seleccionadas_ADP_x])

            #st.write(variables_)
            
            st.subheader("Dataframe de las variables predictoras")
            st.dataframe(X)


            st.subheader("Dataframe de las variable clase")
            Y = np.array(dataframe_AD_Pronostico[select_var_pronostico])
            st.dataframe(Y)



            st.subheader("División del conjunto de datos")

            porcentaje_validacion = st.number_input('Selecciona por favor qué porcentaje del dataset utilizarás para la validación del modelo', min_value=0.01, max_value=0.40, format="%.2f", key='valor_validacion', help='Sólo se admiten valores numéricos positivos entre 0.01 y 0.40', on_change=None, args=None, kwargs=None)
            #st.write(porcentaje_validacion)


            if st.button("Presiona este botón para generar la división del dataset y para contiinuar con el algoritmo", key=915, help=None, on_click=None, args=None, kwargs=None):
                X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = porcentaje_validacion, random_state = 0, shuffle = True)

                st.subheader("Visualización del dataframe de entrenamiento de variables predictoras")
                st.dataframe(X_train)


            
                st.subheader("Visualización del dataframe de entrenamiento de la variable a predecir")
                st.dataframe(Y_train)


                st.subheader("Entrenamiento del modelo (prónóstico generado)")
                st.write("En esta sección podrás apreciar la clasificación generada después de entrenar al modelo. Los resultados provienen del porcentaje del conjunto de datos de validación seleccionado previamente.")




                ClasificacionAD = DecisionTreeClassifier()
                ClasificacionAD.fit(X_train, Y_train)

                Y_Clasificacion = ClasificacionAD.predict(X_validation)
                st.dataframe(Y_Clasificacion)



                st.subheader("Comparación entre las variables clase reales de los datos de validación vs las variables clase predecidas con el modelo")
                st.write("El siguiente dataframe muestra en la primera columna los valores reales del conjunto de datos de validación y en la segunda columna se muestran los valores predecidos con el modelo.")
                Valores = pd.DataFrame(Y_validation, Y_Clasificacion)
                st.dataframe(Valores) 


                st.subheader("Exactitud promedio de la validación")
                #Se calcula la exactitud promedio de la validación
                st.info(ClasificacionAD.score(X_validation, Y_validation))


                st.subheader("Matriz de clasificación")

                Y_Clasificacion = ClasificacionAD.predict(X_validation)
                Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), 
                                                   Y_Clasificacion, 
                                                   rownames=['Real'], 
                                                   colnames=['Clasificación']) 
                
                st.write(Matriz_Clasificacion)


                st.subheader("Reporte de clasificación")

                st.write("Los valores representan las siguientes métricas, en orden de izquierda a derecha: precision, recall, f1-score, support")


                #Reporte de la clasificación
                st.write('Criterio: \n', ClasificacionAD.criterion)
                st.write("Exactitud", ClasificacionAD.score(X_validation, Y_validation))
                st.write(classification_report(Y_validation, Y_Clasificacion))



                st.subheader("Importancia de cada variable predictora del modelo")


                Importancia = pd.DataFrame({'Variable': list(dataframe_AD_Pronostico[variables_seleccionadas_ADP_x]),
                            'Importancia': ClasificacionAD.feature_importances_}).sort_values('Importancia', ascending=False)
                
                st.write(Importancia)



                st.subheader("Visualización del árbol")
                



                graph_911 ,ax= plt.subplots(figsize=(20,20))
                
                
                plot_tree(ClasificacionAD, feature_names = list(dataframe_AD_Pronostico[variables_seleccionadas_ADP_x]))


                st.pyplot(graph_911)                

                st.subheader("Nuevos Pronósticos")
                #st.write(list(dataframe_AD_Pronostico[variables_seleccionadas_ADP_x]))
                
                #listaSSP=list(dataframe_AD_Pronostico[variables_seleccionadas_ADP_x])
                #num_variables_seleccionadas=len(listaSSP)
                #st.write(num_variables_seleccionadas)
                
                #arraySSP= np.array(listaSSP)
                #st.write(arraySSP)


                #Intento de implementación de Sistema Automatizado de Nuevos Pronósticos

                #for i in range(1,num_variables_seleccionadas): #add n rows of data

                    #dataframe_AD_Pronostico.loc[i, [dataframe_AD_Pronostico[i]]] = 0.1
                    #df.loc[i, ['B']] = randint(0,99)
                    #df.loc[i, ['C']] = randint(0,99)
                    #df.loc[i, ['D']] = randint(0,99)
                    #df.loc[i, ['E']] = randint(0,99)
                #st.dataframe(dataframe_AD_Pronostico)

                valor_textura = st.number_input('Introduce el valor de Texture', key='valor_textura_1',value=10.38, format="%.4f")
  

                valor_perimetro = st.number_input('Introduce el valor de Perimeter', key='valor_perimeter_1',value=120.8, format="%.4f")


                valor_Smoothness = st.number_input('Introduce el valor de Smoothness', key='valor_Smoothness_1',value=0.11840, format="%.4f")
                
                valor_Compactness = st.number_input('Introduce el valor de Compactness', key='valor_Compactness_1',value=0.27760, format="%.4f")


                valor_Symmetry = st.number_input('Introduce el valor de Symmetry', key='valor_Symmetry_1',value=0.2419, format="%.4f")

                valor_FractalDimension = st.number_input('Introduce el valor de FractalDimension', key='valor_FractalDimension_1',value=0.07871, format="%.4f")





                AreaTumorID2 = pd.DataFrame({'Texture': [valor_textura], 
                                             'Perimeter': [valor_perimetro], 
                                             'Smoothness': [valor_Smoothness], 
                                             'Compactness': [valor_Compactness], 
                                             'Symmetry': [valor_Symmetry], 
                                             'FractalDimension': [valor_FractalDimension]})

                st.subheader("Datos recopilados de las variables predictoras")
                st.dataframe(AreaTumorID2)

                st.subheader("Pronóstico generado con el modelo")
                df_pronostico=ClasificacionAD.predict(AreaTumorID2)
                st.write('El pronóstico es de:')
                st.info(df_pronostico[0])



            else:
                st.write("")


