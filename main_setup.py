from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
from prophet import Prophet

# 1. Importing your custom modules from the 'src' folder
from src.data_loader import load_and_clean_retail_data
from src.evaluation import calculate_metrics

# 2. Fetch dataset 
print("Step 1: Fetching data from UCI...")
online_retail = fetch_ucirepo(id=352) 
X = online_retail.data.features 

# 3. Clean and Transform Data
# This uses the logic you wrote in src/data_loader.py
print("Step 2: Transforming transactions into Daily Time Series...")
ts_data = load_and_clean_retail_data(X)

# 4. Quick Visualization
print("Step 3: Generating Exploratory Plot...")
ts_data.plot(figsize=(12, 6), title="Daily Sales Revenue - UCI Online Retail")
plt.ylabel("Revenue")
plt.show()

# 5. Forecasting with Prophet
print("Step 4: Training Prophet Model...")
# Prophet needs a DataFrame with columns 'ds' (date) and 'y' (value)
df_prophet = ts_data.reset_index().rename(columns={'InvoiceDate': 'ds', 'Revenue': 'y'})

model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.fit(df_prophet)

# Predict 30 days into the future
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# 6. Show Results
print("Step 5: Displaying Forecast...")
model.plot(forecast)
plt.title("30-Day Sales Forecast")
plt.show()

# 7. Print Components (Trend/Weekly patterns)
model.plot_components(forecast)
plt.show()