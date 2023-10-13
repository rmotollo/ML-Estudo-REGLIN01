from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Carregar o modelo treinado (substitua pelo seu próprio modelo)
with open('./modelos/reglin_cons_cerveja.pkl', 'rb') as f:
    model = pickle.load(f)



# Para fins de demonstração, criaremos um modelo simples aqui

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        feature = data['Temperatura Maxima (C)']
        df = pd.DataFrame({ 'Temperatura Maxima (C)': [feature] })
        prediction = model.predict(df)
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
