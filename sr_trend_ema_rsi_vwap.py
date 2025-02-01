from ib_insync import *
import pandas as pd
import numpy as np
import mplfinance as mpf
from datetime import datetime, timedelta
import asyncio
import nest_asyncio
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Shared Configuration
symbol = "F" # Default Ticker

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler('trading_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize addplots list at the start of your function
addplots = []

def get_historical_data(ib, symbol, exchange='SMART', currency='USD'):
    contract = Stock(symbol, exchange, currency)
    end_time = datetime.now()
    
    # Get data for all three timeframes
    bars_5m = ib.reqHistoricalData(
        contract,
        endDateTime=end_time,
        durationStr='1 D',
        barSizeSetting='5 mins',
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1
    )
    
    bars_30m = ib.reqHistoricalData(
        contract,
        endDateTime=end_time,
        durationStr='1 D',
        barSizeSetting='30 mins',
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1
    )
    
    bars_1h = ib.reqHistoricalData(
        contract,
        endDateTime=end_time,
        durationStr='1 D',
        barSizeSetting='1 hour',
        whatToShow='TRADES',
        useRTH=True,
        formatDate=1
    )
    
    # Convert to DataFrames
    df_5m = util.df(bars_5m)
    df_30m = util.df(bars_30m)
    df_1h = util.df(bars_1h)
    
    # Add volume analysis indicators
    for df in [df_5m, df_30m, df_1h]:
        # Calculate VWAP
        df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        # Calculate Volume Moving Average
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        
        # Calculate relative volume (current volume vs average)
        df['relative_volume'] = df['volume'] / df['volume_ma']
        
        # Add volume trend analysis
        df['volume_trend'] = df.apply(lambda x: 1 if x['close'] > x['open'] and x['volume'] > x['volume_ma'] 
                                    else -1 if x['close'] < x['open'] and x['volume'] > x['volume_ma']
                                    else 0, axis=1)
    
    df_5m.set_index('date', inplace=True)
    df_30m.set_index('date', inplace=True)
    df_1h.set_index('date', inplace=True)
    
    # Enhanced logging with volume analysis
    for index, row in df_5m.iterrows():
        logger.info(
            f"5M | Symbol: {symbol} | Time: {index} | "
            f"Open: {row['open']:.2f} | High: {row['high']:.2f} | "
            f"Low: {row['low']:.2f} | Close: {row['close']:.2f} | "
            f"Volume: {row['volume']} | VWAP: {row['VWAP']:.2f} | "
            f"Rel Volume: {row['relative_volume']:.2f}"
        )
        
        # Find corresponding 30m candle
        thirty_min_time = index.replace(minute=(index.minute // 30) * 30)
        if thirty_min_time in df_30m.index:
            thirty_min_data = df_30m.loc[thirty_min_time]
            logger.info(
                f"30M | Symbol: {symbol} | Time: {thirty_min_time} | "
                f"Open: {thirty_min_data['open']:.2f} | High: {thirty_min_data['high']:.2f} | "
                f"Low: {thirty_min_data['low']:.2f} | Close: {thirty_min_data['close']:.2f} | "
                f"Volume: {thirty_min_data['volume']} | VWAP: {thirty_min_data['VWAP']:.2f}"
            )
        
        # Find corresponding 1h candle
        hour_time = index.replace(minute=0)
        if hour_time in df_1h.index:
            hour_data = df_1h.loc[hour_time]
            logger.info(
                f"1H | Symbol: {symbol} | Time: {hour_time} | "
                f"Open: {hour_data['open']:.2f} | High: {hour_data['high']:.2f} | "
                f"Low: {hour_data['low']:.2f} | Close: {hour_data['close']:.2f} | "
                f"Volume: {hour_data['volume']} | VWAP: {hour_data['VWAP']:.2f}"
            )
    return df_5m, df_30m, df_1h

def find_support_resistance(df_5m, df_15m, df_30m, window=20, min_touches=3, price_threshold=0.15):
    levels = {'support': [], 'resistance': []}
    touches = {}
    
    # Calculate swing highs and lows for all timeframes
    df_5m['swing_high'] = df_5m['high'].rolling(window=5, center=True).max()
    df_5m['swing_low'] = df_5m['low'].rolling(window=5, center=True).min()
    
    df_15m['swing_high'] = df_15m['high'].rolling(window=3, center=True).max()
    df_15m['swing_low'] = df_15m['low'].rolling(window=3, center=True).min()
    
    df_30m['swing_high'] = df_30m['high'].rolling(window=2, center=True).max()
    df_30m['swing_low'] = df_30m['low'].rolling(window=2, center=True).min()
    
    # Get potential levels from higher timeframes
    higher_tf_levels = set()
    
    # Add 15m levels
    for _, row in df_15m.iterrows():
        higher_tf_levels.add(row['swing_high'])
        higher_tf_levels.add(row['swing_low'])
    
    # Add 30m levels
    for _, row in df_30m.iterrows():
        higher_tf_levels.add(row['swing_high'])
        higher_tf_levels.add(row['swing_low'])
    
    # Process 5m data with confluence from higher timeframes
    price_history = []
    for i in range(len(df_5m)):
        current_price = df_5m.iloc[i]
        price_history.append({
            'high': current_price['high'],
            'low': current_price['low'],
            'close': current_price['close'],
            'swing_high': current_price['swing_high'],
            'swing_low': current_price['swing_low']
        })
        
        if len(price_history) >= window:
            window_prices = price_history[-window:]
            
            if i % 3 == 0:
                current_high = window_prices[-1]['swing_high']
                current_low = window_prices[-1]['swing_low']
                
                # Check for confluence with higher timeframe levels
                has_higher_tf_confluence_high = any(abs(level - current_high)/current_high < price_threshold 
                                                  for level in higher_tf_levels)
                has_higher_tf_confluence_low = any(abs(level - current_low)/current_low < price_threshold 
                                                 for level in higher_tf_levels)
                
                support_count = 0
                resistance_count = 0
                
                for k in range(len(window_prices) - 1):
                    # Support level logic with confluence
                    if abs(window_prices[k]['low'] - current_low) / current_low < price_threshold:
                        if (window_prices[k+1]['close'] > current_low and 
                            window_prices[k+1]['low'] > window_prices[k]['low'] and
                            has_higher_tf_confluence_low):
                            support_count += 1
                    
                    # Resistance level logic with confluence
                    if abs(window_prices[k]['high'] - current_high) / current_high < price_threshold:
                        if (window_prices[k+1]['close'] < current_high and 
                            window_prices[k+1]['high'] < window_prices[k]['high'] and
                            has_higher_tf_confluence_high):
                            resistance_count += 1
                
                # Add levels with strong confirmation and higher timeframe confluence
                if support_count >= min_touches:
                    levels['support'].append(current_low)
                    touches[current_low] = support_count
                
                if resistance_count >= min_touches:
                    levels['resistance'].append(current_high)
                    touches[current_high] = resistance_count
    
    # Improved level filtering
    for level_type in ['support', 'resistance']:
        levels[level_type] = sorted(set(levels[level_type]))
        filtered = []
        
        # Group nearby levels
        current_group = []
        for price in levels[level_type]:
            if not current_group or abs(price - current_group[-1])/current_group[-1] <= 0.005:
                current_group.append(price)
            else:
                # Take the most touched level from the group
                best_level = max(current_group, key=lambda x: touches[x])
                filtered.append(best_level)
                current_group = [price]
        
        if current_group:
            best_level = max(current_group, key=lambda x: touches[x])
            filtered.append(best_level)
        
        levels[level_type] = filtered
    
    return levels, touches

def get_color_intensity(touch_count, min_touches, max_touches):
    intensity = 0.5 + 0.5 * ((touch_count - min_touches) / (max_touches - min_touches))
    return min(max(intensity, 0.5), 1.0)

def find_trendlines(df_5m, min_touches=3):
    trendlines = {'bullish': [], 'bearish': []}
    
    # Create lists to store potential trend points
    bullish_sequences = []
    bearish_sequences = []
    current_bullish = []
    current_bearish = []
    
    # Helper function to check if a candle is green/red
    def is_green_candle(row):
        return row['close'] > row['open']
    
    def is_red_candle(row):
        return row['close'] < row['open']
    
    # Iterate through the dataframe
    for i in range(len(df_5m)):
        current_row = df_5m.iloc[i]
        
        # Check for bullish sequence
        if is_green_candle(current_row):
            if not current_bullish:
                current_bullish = [(df_5m.index[i], current_row['close'])]
            else:
                if current_row['close'] > current_bullish[-1][1]:
                    current_bullish.append((df_5m.index[i], current_row['close']))
                else:
                    if len(current_bullish) >= min_touches:
                        bullish_sequences.append(current_bullish)
                    current_bullish = [(df_5m.index[i], current_row['close'])]
        
        # Check for bearish sequence
        if is_red_candle(current_row):
            if not current_bearish:
                current_bearish = [(df_5m.index[i], current_row['close'])]
            else:
                if current_row['close'] < current_bearish[-1][1]:
                    current_bearish.append((df_5m.index[i], current_row['close']))
                else:
                    if len(current_bearish) >= min_touches:
                        bearish_sequences.append(current_bearish)
                    current_bearish = [(df_5m.index[i], current_row['close'])]
    
    # Create trend lines from sequences
    for sequence in bullish_sequences:
        if len(sequence) >= min_touches:
            trend_line = pd.Series(index=df_5m.index, dtype=float)
            start_date, start_price = sequence[0]
            end_date, end_price = sequence[-1]
            
            # Calculate line gradient
            time_delta = (end_date - start_date).total_seconds()
            price_delta = end_price - start_price
            gradient = price_delta / time_delta
            
            # Create trend line points
            for date in df_5m.index:
                if start_date <= date <= end_date:
                    seconds_from_start = (date - start_date).total_seconds()
                    trend_line[date] = start_price + (gradient * seconds_from_start)
            
            trendlines['bullish'].append(trend_line)
    
    for sequence in bearish_sequences:
        if len(sequence) >= min_touches:
            trend_line = pd.Series(index=df_5m.index, dtype=float)
            start_date, start_price = sequence[0]
            end_date, end_price = sequence[-1]
            
            # Calculate line gradient
            time_delta = (end_date - start_date).total_seconds()
            price_delta = end_price - start_price
            gradient = price_delta / time_delta
            
            # Create trend line points
            for date in df_5m.index:
                if start_date <= date <= end_date:
                    seconds_from_start = (date - start_date).total_seconds()
                    trend_line[date] = start_price + (gradient * seconds_from_start)
            
            trendlines['bearish'].append(trend_line)
    
    return trendlines

def plot_candlestick(df_5m, df_15m, df_30m):
    addplots = []
    
    if df_5m.empty:
        logger.warning("No data available for plotting")
        return

    # Calculate RSI
    delta = df_5m['close'].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()  # EMA alternative
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df_5m['RSI'] = 100 - (100 / (1 + rs))

    # Create RSI plot
    rsi_plot = mpf.make_addplot(
        df_5m['RSI'],
        panel=1,
        color='purple',
        width=0.7,
        ylabel='RSI',
        ylim=(0, 100),
        alpha=0.8
    )
    addplots.append(rsi_plot)

    # Calculate EMAs
    df_5m['5_EMA'] = df_5m['close'].ewm(span=5, adjust=False).mean()
    df_5m['10_EMA'] = df_5m['close'].ewm(span=10, adjust=False).mean()
    df_5m['20_EMA'] = df_5m['close'].ewm(span=20, adjust=False).mean()

    # Add EMAs to addplots
    addplots.append(mpf.make_addplot(
        df_5m['5_EMA'],
        color='blue',
        width=1.5,
        alpha=0.8,
        secondary_y=False
    ))
    
    addplots.append(mpf.make_addplot(
        df_5m['10_EMA'],
        color='orange',
        width=1.5,
        alpha=0.8,
        secondary_y=False
    ))

    addplots.append(mpf.make_addplot(
        df_5m['20_EMA'],
        color='red',
        width=1.5,
        alpha=0.8,
        secondary_y=False
    ))

    # Calculate VWAP
    df_5m['VWAP'] = (df_5m['volume'] * (df_5m['high'] + df_5m['low'] + df_5m['close']) / 3).cumsum() / df_5m['volume'].cumsum()

    # Add VWAP to addplots
    addplots.append(mpf.make_addplot(
        df_5m['VWAP'],
        color='purple',
        width=1.5,
        alpha=0.8,
        secondary_y=False
    ))
        
    # Get support/resistance levels
    min_touches = 5
    levels, touches = find_support_resistance(df_5m, df_15m, df_30m, min_touches=min_touches)
    
    # Get trendlines
    trendlines = find_trendlines(df_5m, min_touches=5)
    
    # Support/Resistance colors
    support_colors = [(0, 0, 1, 1.0) for _ in levels['support']]
    resistance_colors = [(1, 0, 0, 1.0) for _ in levels['resistance']]
    
    hlines = dict(
        hlines=levels['support'] + levels['resistance'],
        colors=support_colors + resistance_colors,
        linestyle='--',
        linewidths=1.0
    )
    
    # Create market colors and style (modified volume colors)
    mc = mpf.make_marketcolors(
        up='green',
        down='red',
        edge='inherit',
        wick='inherit',
        volume={'up':'green', 'down':'red'}  # Volume colors moved here
    )
    
    s = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle='dotted'
    )
    
    # Add trendlines
    for line in trendlines['bullish']:
        addplots.append(mpf.make_addplot(
            line,
            color='yellow',
            width=1.5,
            alpha=1.0,
            linestyle='-',
            secondary_y=False
        ))

    for line in trendlines['bearish']:
        addplots.append(mpf.make_addplot(
            line,
            color='yellow',
            width=1.5,
            alpha=1.0,
            linestyle='-',
            secondary_y=False
        ))

    # Create custom legend handles
    legend_handles = [
        Patch(facecolor="green", label="Bullish Candle"),
        Patch(facecolor="red", label="Bearish Candle"),
        Patch(facecolor="yellow", label="Trendlines"),
        Patch(facecolor="blue", label="Support"),
        Patch(facecolor="red", label="Resistance"),
        Patch(facecolor="blue", label="5 EMA"),
        Patch(facecolor="orange", label="10 EMA"),
        Patch(facecolor="red", label="20 EMA"),
        Patch(facecolor="purple", label="VWAP")
    ]

    # Create figure and axis objects with adjusted layout
    fig, axlist = mpf.plot(
        df_5m,
        type='candle',
        style=s,
        title='',
        volume=True,  # Changed from dict to boolean
        volume_panel=2,  # Added separate volume_panel parameter
        addplot=addplots,
        hlines=hlines,
        figsize=(12, 8),
        panel_ratios=(6, 2, 2),
        returnfig=True
    )

    # Configure RSI panel
    ax_rsi = axlist[1]
    ax_rsi.grid(True, which='both', linestyle='--', alpha=0.5)
    ax_rsi.set_yticks([30, 50, 70])
    ax_rsi.axhline(30, color='gray', linestyle='--', alpha=0.7)
    ax_rsi.axhline(70, color='gray', linestyle='--', alpha=0.7)

    # Add title above the chart
    fig.suptitle('5-Minute Candlestick Chart with Multi-Timeframe S/R Levels\nStock | $' + symbol, y=0.98, fontsize=14)

    # Add legend above the chart
    ax = axlist[0]
    ax.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.12),
        ncol=5,
        frameon=True,
        fancybox=True,
        shadow=True
    )

    # Adjust layout to prevent cutoff
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Log levels
    logger.info("Support Levels (from strongest):")
    for price in sorted(levels['support'], key=lambda x: touches[x], reverse=True):
        logger.info(f"Level: {price:.2f}, Strength: {touches[price]} touches")
    
    logger.info("Resistance Levels (from strongest):")
    for price in sorted(levels['resistance'], key=lambda x: touches[x], reverse=True):
        logger.info(f"Level: {price:.2f}, Strength: {touches[price]} touches")

def main():
    global symbol
    # Connect to TWS
    ib = IB()
    symbol = 'F' # Default Ticker
    ib.connect('127.0.0.1', 7497, clientId=1)

    logger.info(f"Starting data collection for {symbol}")
    
    try:
        df_5m, df_15m, df_30m = get_historical_data(ib, symbol)
        plot_candlestick(df_5m, df_15m, df_30m)
        logger.info(f"Successfully completed data collection and plotting for {symbol}")
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
    finally:
        ib.disconnect()
        logger.info("Disconnected from TWS")

if __name__ == "__main__":
    main()