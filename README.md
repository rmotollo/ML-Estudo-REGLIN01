---
# Projeto de Regressão Linear para Previsão de Consumo de Cerveja 🍻

## Descrição

Este projeto utiliza um modelo de regressão linear para prever o consumo de cerveja com base em diferentes variáveis, como temperatura máxima, precipitação e se é final de semana ou não. O projeto também inclui uma API Flask para fazer previsões em tempo real e uma interface Streamlit para visualização de dados.

## Requisitos

- Python 3.x
- Flask
- scikit-learn
- pandas
- matplotlib
- seaborn
- Streamlit

## Instalação

1. Clone este repositório
   ```
   git clone https://github.com/seu_usuario/seu_repositorio.git
   ```
2. Entre no diretório
   ```
   cd seu_repositorio
   ```
3. Instale as dependências
   ```
   pip install -r requirements.txt
   ```

## Uso

### Treinamento do Modelo

Execute o script `reglin_cons_cerveja.py` para treinar o modelo.

```
python reglin_cons_cerveja.py
```

### Executando a API

Execute o script `app.py` para iniciar a API Flask.

```
python app.py
```

A API estará disponível em `http://127.0.0.1:5000/`.

### Visualização de Dados com Streamlit

Para iniciar a interface Streamlit, execute o seguinte comando:

```
streamlit run main.py
```

Isso abrirá uma nova janela no navegador onde você pode interagir com os dados e visualizações.

## Endpoints

### POST /predict

Envie um JSON com a temperatura máxima para receber uma previsão de consumo de cerveja.

Exemplo de requisição:

```json
{
  "Temperatura Maxima (C)": 32.5
}
```

## Contribuição

Sinta-se à vontade para contribuir com o projeto. Abra uma issue ou envie um pull request.

## Licença

None

---




