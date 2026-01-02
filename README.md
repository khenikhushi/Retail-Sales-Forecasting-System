# Time-Series Revenue Forecasting

A time-series forecasting project to predict daily revenue using historical online retail transaction data. The project compares classical statistical models with modern forecasting techniques to analyze trends, seasonality, and revenue patterns.

---

## 📌 Project Overview
- Dataset: UCI Online Retail Dataset (500k+ transactions)
- Goal: Forecast daily revenue using time-series models
- Models used: ARIMA and Facebook Prophet

---

## 🛠 Tech Stack
- Python  
- Pandas, NumPy  
- Statsmodels  
- Scikit-learn  
- Facebook Prophet  
- Matplotlib  

---

## 📊 Methodology
1. Converted transactional data into a daily revenue time series  
2. Performed trend and weekly seasonality analysis using time-series decomposition  
3. Applied ADF test for stationarity and used log differencing (d=1)  
4. Built and tuned an ARIMA model for revenue forecasting  
5. Developed a Prophet model with weekly seasonality and UK holiday effects  
6. Evaluated models using RMSE and MAPE with time-series cross-validation  

---

## 📈 Results
- Captured clear weekly seasonality and long-term revenue trends  
- Prophet model showed robustness to outliers and holiday effects  
- Model performance evaluated using RMSE and MAPE metrics  

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/khenikhushi/time-series-revenue-forecasting.git
