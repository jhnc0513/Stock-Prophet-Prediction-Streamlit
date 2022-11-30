import streamlit as st
import datetime as dt
from datetime import date, datetime, tzinfo
import yfinance as yf 
from prophet import Prophet 
from prophet.plot import plot_plotly 
from plotly import graph_objs as go
import pandas as pd 

START = "2015-01-01"
TODAY = datetime.now().date()


st.title("Stock Prediction App")

stocks = ('AAPL','GOOGL','MSFT','GME','AMZN','TSLA','BABA','BTC-USD','ETH-USD')
selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction", 1, 4)
period = n_years * 365

@st.cache
def load_data(ticker):
    data = yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].dt.date
    return data 

data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done!")

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name ='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name ='stock_close'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

df_train = data[['Date', 'Close']]

new_names = {
    "Date": "ds",
    "Close": "y",
}
df_train = df_train.rename(columns=new_names)
df_train['ds'] = pd.to_datetime(df_train['ds'])
df_train['ds'] = df_train['ds'].dt.date

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

forecast_table = m.predict(future)
forecast_table['ds'] = forecast_table['ds'].dt.date


st.subheader('Forecast data')
st.write(forecast_table.tail())

st.write('forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)
