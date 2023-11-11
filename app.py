import streamlit as st
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
from fbprophet.plot import plot_plotly

st.title("Análise e Previsão de Vendas")

# Carregar dados históricos de vendas
@st.cache
def load_data():
    data = pd.read_csv('dados_vendas_sazonalidade.csv')
    return data

data = load_data()

# Renomear as colunas para o formato exigido pelo Prophet
data.rename(columns={'Data': 'ds', 'Vendas': 'y'}, inplace=True)

# Carregar dados de feriados
feriados = pd.read_csv('feriados.csv')
feriados.rename(columns={'Data': 'ds', 'Feriados': 'holiday'}, inplace=True)

# Exibir dados históricos
st.subheader("Dados Históricos de Vendas")
st.write(data)

# Criar modelo Prophet com feriados
model = Prophet(holidays=feriados, yearly_seasonality=True)
model.add_country_holidays(country_name='BR')  # Adicione feriados brasileiros, se aplicável
model.fit(data)

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
