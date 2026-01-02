import sys
import os
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo  # Added this
from prophet import Prophet

# 1. Fix the Path (so it can see the 'src' folder)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 2. Import your custom data loader
from src.data_loader import load_and_clean_retail_data

# 3. FETCH AND DEFINE 'ts' (This was missing!)
print("Fetching and processing data for Prophet...")
online_retail = fetch_ucirepo(id=352) 
X = online_retail.data.features 
ts = load_and_clean_retail_data(X)  # Now 'ts' is defined!

# 4. Prepare data for Prophet
# Prophet requires columns 'ds' and 'y'
df_prophet = ts.reset_index().rename(columns={'InvoiceDate': 'ds', 'Revenue': 'y'})

# 5. Initialize and Train the Model
print("Training Prophet model...")
model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model.add_country_holidays(country_name='UK')
model.fit(df_prophet)

# 6. Forecast for the next 30 days
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# 7. Plotting the results
print("Generating Forecast Plots...")
model.plot(forecast)
plt.title("30-Day Revenue Forecast (Prophet)")
plt.show()

# Optional: Show the Trend and Weekly patterns
model.plot_components(forecast)
plt.show()