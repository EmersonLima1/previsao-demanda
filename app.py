import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.plot import plot_plotly

st.title("Análise e Previsão de Vendas")

# Carregar dados históricos de vendas
@st.cache
def load_data():
    data = pd.read_csv('dados_vendas_sazonalidade.csv')
    return data

data = load_data()

# Exibir dados históricos
st.subheader("Dados Históricos de Vendas")
st.write(data)

# Criar modelo Prophet
model = Prophet(holidays=data[data['Feriados'] == 1], yearly_seasonality=True)
model.fit(data[['Data', 'Vendas']].rename(columns={'Data': 'ds', 'Vendas': 'y'}))

# Prever vendas futuras
st.subheader("Previsão de Vendas Futuras")
future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)

# Plotar previsão
fig = plot_plotly(model, forecast)
st.plotly_chart(fig)

# Plotar componentes da previsão
st.subheader("Componentes da Previsão")
fig = model.plot_components(forecast)
st.write(fig)

# Exportar previsão para CSV
st.subheader("Exportar Previsão para CSV")
forecast.to_csv('previsao_vendas.csv', index=False)

st.write("Previsão exportada para 'previsao_vendas.csv'")
