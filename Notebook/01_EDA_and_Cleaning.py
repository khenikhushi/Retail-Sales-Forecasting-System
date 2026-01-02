# import sys
# import os
# # This adds the root directory (sale) to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from ucimlrepo import fetch_ucirepo 
# from src.data_loader import load_and_clean_retail_data
# from statsmodels.tsa.seasonal import seasonal_decompose
# import matplotlib.pyplot as plt



# # Fetch Data
# online_retail = fetch_ucirepo(id=352) 
# X = online_retail.data.features 
# ts = load_and_clean_retail_data(X)

# # Visualizing the components
# # 
# result = seasonal_decompose(ts, model='additive', period=7)
# result.plot()
# plt.show()








from ucimlrepo import fetch_ucirepo 
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import pandas as pd

# 1. Fetch data directly from UCI Repository
online_retail = fetch_ucirepo(id=352) 
df = online_retail.data.features 

# 2. Basic Preprocessing (Condensed)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Revenue'] = df['Quantity'] * df['UnitPrice']
ts = df.set_index('InvoiceDate')['Revenue'].resample('D').sum().fillna(0)

# 3. Decompose and Plot
# seasonal_decompose(ts, model='additive', period=7).plot()
# plt.show()

result = seasonal_decompose(ts, model='additive', period=7)
fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

# Plot Observed with a 7-day rolling mean for clarity
axes[0].plot(ts, label='Original', color='royalblue', alpha=0.5)
axes[0].plot(ts.rolling(window=7).mean(), label='7-Day Rolling Mean', color='crimson', linewidth=2)
axes[0].set_title('Daily Revenue Analysis', fontsize=16, fontweight='bold')
axes[0].legend()

# Plot Trend
axes[1].plot(result.trend, color='forestgreen', linewidth=2)
axes[1].set_title('Long-Term Sales Trend', fontsize=14)

# Plot Seasonal (Highlighting the 7-day cycle)
axes[2].plot(result.seasonal, color='darkorange', linewidth=2)
axes[2].set_title('Weekly Seasonality Pattern', fontsize=14)

# Plot Residuals (Using a scatter plot to identify outliers)
axes[3].scatter(ts.index, result.resid, color='purple', s=10, alpha=0.6)
axes[3].axhline(0, color='black', linestyle='--')
axes[3].set_title('Residuals (Anomalies & Noise)', fontsize=14)

plt.tight_layout()
plt.show()

