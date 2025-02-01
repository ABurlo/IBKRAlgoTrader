from ib_insync import *
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nest_asyncio
from sr_trend_ema_rsi_vwap import symbol

nest_asyncio.apply()

def fetch_historical_data(symbols, ib):
    """Fetch historical data for a list of symbols."""
    dataframes = {}
    for symbol in symbols:
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            bars = ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr='30 D',
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True
            )
            df = util.df(bars)
            df.set_index('date', inplace=True)
            dataframes[symbol] = df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
    return dataframes

def plot_correlation_heatmap(primary_stock, other_stocks):
    """Plot correlation heatmap separately."""
    returns = pd.DataFrame()
    primary_symbol = primary_stock.name
    returns[primary_symbol] = primary_stock['close'].pct_change()
    
    for symbol, data in other_stocks.items():
        returns[symbol] = data['close'].pct_change()
    
    returns.dropna(inplace=True)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        returns.corr(),
        annot=True,
        cmap='coolwarm',
        center=0,
        cbar_kws={'label': 'Correlation Coefficient'}
    )
    plt.title(f'Correlation Heatmap: {primary_symbol} vs Others')
    plt.show()

def plot_normalized_prices(primary_stock, other_stocks):
    """Plot all normalized price movements on one chart."""
    plt.figure(figsize=(12, 6))
    
    # Plot primary stock
    primary_symbol = primary_stock.name
    primary_normalized = primary_stock['close'] / primary_stock['close'].iloc[0]
    plt.plot(primary_normalized, label=primary_symbol, linewidth=2.5, linestyle='--')
    
    # Plot other stocks
    for symbol, data in other_stocks.items():
        normalized_price = data['close'] / data['close'].iloc[0]
        plt.plot(normalized_price, label=symbol, alpha=0.8)
    
    plt.title('Normalized Price Movements (30 Days)')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
symbols = [symbol, 'AAPL', 'MSFT', 'GOOG']  # <-- Modify symbols here
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

try:
    # Fetch historical data
    dataframes = fetch_historical_data(symbols, ib)
    
    # Define primary stock and other stocks
    primary_symbol = symbols[0]
    primary_data = dataframes[primary_symbol].copy()
    primary_data.name = primary_symbol
    other_stocks = {sym: df for sym, df in dataframes.items() if sym != primary_symbol}

    # Plot heatmap and normalized price movements separately
    plot_correlation_heatmap(primary_data, other_stocks)
    plot_normalized_prices(primary_data, other_stocks)

finally:
    ib.disconnect()