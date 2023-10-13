import streamlit as st
import reglin_cons_cerveja as rcc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Importing the pages


@st.cache_data
def load_data():
    df = pd.read_csv("./dados/Consumo_cerveja.csv", sep=",")
    df = pd.DataFrame(df)
    #converter strings para float
    for col in ['Temperatura Media (C)', 'Temperatura Minima (C)', 'Temperatura Maxima (C)', 'Precipitacao (mm)']:
        df[col] = df[col].str.replace(',', '.').astype(np.float64)

    df.dropna(subset=['Final de Semana'], inplace=True)
    df['Final de Semana'] = df['Final de Semana'].astype(np.int8)

    return df

df = load_data()

st.sidebar.title("Menu")
page = st.sidebar.selectbox("Escolha uma página", ["Página Inicial", "Análise Exploratória", "Regressão Linear" ])

if page == "Página Inicial":
    st.title("Página Inicial")
    st.write("Esta é a página inicial")


#Analise exploratória
if page == "Análise Exploratória":
    st.title('Consumo de cerveja')
    st.subheader('Análise de regressão linear')
    st.dataframe(df.head())
    # Seletores
    option_x = st.selectbox('Escolha uma variável para o eixo X:', df.columns)
    option_y = st.selectbox('Escolha uma variável para o eixo Y:', df.columns)
    option_color = st.selectbox('Escolha uma variável para a cor:', df.columns)

    # Gráfico de Dispersão
    st.markdown('### Gráfico de Dispersão')
    fig1 = px.scatter(df, x=option_x, y=option_y, color=option_color)
    st.plotly_chart(fig1)

    # Gráfico de Barras
    st.markdown('### Gráfico de Barras')
    fig2 = px.bar(df, x=option_x, y=option_y)
    st.plotly_chart(fig2)

    # Tabela de Dados
    if st.checkbox('Mostrar tabela de dados'):
        st.write(df)

    # Estatísticas Básicas
    if st.checkbox('Mostrar estatísticas básicas'):
        st.write(df.describe())

    if st.checkbox('Mostrar Heatmap de Correlação'):
        st.write('Heatmap de Correlação')
        num_col = df.select_dtypes(include = [float])
        corr_matrix = num_col.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        plt.title('Heatmap de Correlação')
        st.pyplot(fig)

if page == "Regressão Linear":
    df_no_data = df.drop('Data', axis=1)
    feature = st.selectbox('Escolha a variável independente:', df_no_data.columns)
    target = st.selectbox('Escolha a variável dependente:', df_no_data.columns)

    # Dividir dados treino e teste
    X = df_no_data[[feature]]
    y = df_no_data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Fazer previsões
    y_pred = model.predict(X_test)

    # Avaliar modelo
    range_de_dados = df[target].max() - df[target].min()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    proporcao = (rmse/range_de_dados)*100


    # Mostrar métricas
    st.write(f'Root Mean Squared Error: {rmse}')
    st.write(f'Proporção de erro: {proporcao:.2f}%')

    # Plotar gráfico
    if st.checkbox('Mostrar gráfico Regressão Linear com teste e treino setos'):
        fig, ax = plt.subplots()  # Crie uma figura e um eixo
        ax.scatter(X_test, y_test, color='red')
        ax.plot(X_test, y_pred, color='blue')
        ax.set_title('Regressão Linear')
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
        st.pyplot(fig)

    user_input = st.number_input(f'Insira um valor para {feature} para prever {target}:')
    user_input = pd.DataFrame({feature: [user_input]})
    user_pred = model.predict(user_input)
    st.write(f'A previsão para {target} é: {user_pred[0]:.2f}')





