import sys
import os
from ucimlrepo import fetch_ucirepo 
import numpy as np

# 1. Fix the Path (so it can see the 'src' folder)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 2. Imports
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from src.data_loader import load_and_clean_retail_data # Added this
from src.evaluation import calculate_metrics

# 3. Fetch and Clean
online_retail = fetch_ucirepo(id=352) 
X = online_retail.data.features 
ts = load_and_clean_retail_data(X)

# NEW: Apply Log Transform to reduce the impact of outliers
# We add 1 to avoid log(0)
ts_log = np.log1p(ts)

# 4. Train/Test Split
train, test = ts_log[:-30], ts_log[-30:]

# 5. Fit ARIMA with d=1 (because p-value was high)
# (p=5, d=1, q=1)
model = ARIMA(train, order=(5, 1, 1))
model_fit = model.fit()

# 6. Predict and Transform back (Exp)
forecast_log = model_fit.forecast(steps=30)
predictions = np.expm1(forecast_log) # Convert back from log scale
actual_test = np.expm1(test)         # Convert back from log scale

# 7. Evaluate
print("\n--- Improved Evaluation Metrics ---")
print(calculate_metrics(actual_test, predictions))