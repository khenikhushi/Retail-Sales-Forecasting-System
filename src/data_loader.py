import pandas as pd

def load_and_clean_retail_data(X_df):
    """
    Transforms raw UCI Online Retail features into a Daily Time Series.
    """
    df = X_df.copy()
    
    # Convert InvoiceDate to datetime objects
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Calculate Revenue: Quantity * UnitPrice
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    
    # Filter out cancelled orders (Quantities < 0) and zero prices
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    
    # Drop rows with missing dates or revenue
    df = df.dropna(subset=['InvoiceDate', 'Revenue'])
    
    # Resample to Daily ('D') frequency and sum the revenue
    # This creates the actual Time Series
    daily_series = df.set_index('InvoiceDate')['Revenue'].resample('D').sum()
    
    # Fill gaps (e.g., days with no sales) using linear interpolation
    daily_series = daily_series.interpolate(method='linear')
    
    return daily_series