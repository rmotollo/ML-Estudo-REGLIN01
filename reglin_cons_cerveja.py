
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#Importando o dataset
dataset = pd.read_csv('./dados/Consumo_cerveja.csv', sep=',')
df = pd.DataFrame(dataset)

#converter strings para float
for col in ['Temperatura Media (C)', 'Temperatura Minima (C)', 'Temperatura Maxima (C)', 'Precipitacao (mm)']:
    df[col] = df[col].str.replace(',', '.').astype(np.float64)
#limpar os dados nulos
df.dropna(subset=['Final de Semana'], inplace=True)
#converter string para inteiro booleano para esse caso
df['Final de Semana'] = df['Final de Semana'].astype(np.int8)

#Preparando os dados para o modelo
#drop em dados não utilizados
df.drop(['Data'], axis=1, inplace=True)

#Dividir dados

X = df[['Temperatura Maxima (C)']]
y = df['Consumo de cerveja (litros)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo
model = LinearRegression()
model.fit(X_train, y_train)

with open('./modelos/reglin_cons_cerveja.pkl', 'wb') as f:
    pickle.dump(model, f)



