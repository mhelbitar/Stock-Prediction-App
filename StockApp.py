import streamlit as st 
import yfinance as yf 
import pandas as pd 
import cufflinks as cf 
import datetime
from prophet import Prophet
from prophet.plot import plot_plotly 
from plotly import graph_objects as go 

# Set the theme
st.set_page_config(layout="wide", page_title="Stock Price App", page_icon="📈")

#Application Heading 
st.markdown('''
# 📈 Stock Price Prediction App 📉
Welcome to the Stock Price Prediction App! This app allows you to explore the historical stock price data for various companies and generate price forecasts using Facebook Prophet.

Simply select a company from the dropdown menu on the left sidebar, choose your desired date range, and adjust the slider to specify the number of years for the price prediction. Let's get started!

**Tools**

            -Built in Python using streamlit, yfinance, cufflinks, pandas, datetime and prophet.
''')

st.write('---')

#Sidebar menu 
st.sidebar.subheader('Query Parameters')
start_date = st.sidebar.date_input("Start date", datetime.date(2015,1,1))
end_date = st.sidebar.date_input("End date",datetime.date(2024,2,28))

#Retriving stock data 
ticker_list = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')
tickerSymbol = st.sidebar.selectbox('Stock ticker', ticker_list) # Select ticker symbol
tickerData = yf.Ticker(tickerSymbol) # Get ticker data
tickerDf = tickerData.history(period='1d', start=start_date, end=end_date) #get the historical prices for this ticker



#Stock Info
string_name = tickerData.info['longName']
st.header('**%s**' % string_name)

string_price = tickerData.info['currentPrice']
st.write(f"Current Price: ${string_price:.2f}")


string_sector = tickerData.info['sector']
st.write("Sector: " + string_sector)


string_summary = tickerData.info['longBusinessSummary']
st.info(string_summary)


#Ticker Data 
st.header('**Ticker data**')
st.write(tickerDf)

# Bollinger bands
st.header('**Bollinger Bands**')
qf=cf.QuantFig(tickerDf,title='First Quant Figure',legend='top',name='GS')
qf.add_bollinger_bands()
fig = qf.iplot(asFigure=True)
st.plotly_chart(fig)



#Stock Price Prediction 

#header
st.header("Price Prediction")

#prediction sidebar slider  
n_years = st.sidebar.slider("Years of prediction", 1, 4)
period = n_years *365

#Forcasting 
df_train = tickerDf.reset_index()[['Date', 'Close']]
df_train['Date'] = pd.to_datetime(df_train['Date']).dt.tz_localize(None)  # Convert to timezone-naive datetime
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

model = Prophet()
model.fit(df_train)

future = model.make_future_dataframe(periods = period)
forecast = model.predict(future)

st.subheader('Forecast Data')
st.write(forecast.tail())

#ploting forcast data 

st.write("Forcast Plot")
plot_plotly(model, forecast)
forcast_figure = plot_plotly(model, forecast)
st.plotly_chart(forcast_figure)

st.write("Forcast components")
forcast_components_figure = model.plot_components(forecast)
st.write(forcast_components_figure)


# Credits to Data Professor
st.markdown('''
### Credits
This application was built with inspiration and guidance from [Data Professor](https://www.youtube.com/channel/UCV8e2g4IWQqK71bbzGDEI4Q). 
Thank you to Data Professor for providing educational content and resources in the field of data science and visualization.
''')