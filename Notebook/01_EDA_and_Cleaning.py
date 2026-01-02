import sys
import os
# This adds the root directory (sale) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ucimlrepo import fetch_ucirepo 
from src.data_loader import load_and_clean_retail_data
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt



# Fetch Data
online_retail = fetch_ucirepo(id=352) 
X = online_retail.data.features 
ts = load_and_clean_retail_data(X)

# Visualizing the components
# 
result = seasonal_decompose(ts, model='additive', period=7)
result.plot()
plt.show()


