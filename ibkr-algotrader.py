# =====================
# CORE IMPORTS
# =====================
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
import yfinance as yf  # Added Yahoo Finance import
from time import sleep  # ADD THIS IMPORT

# =====================
# GLOBAL CONFIGURATION
# =====================
symbol = "CVNA"  # Default Ticker
initial_balance = 10000  # IBKR C.B.U. initial balance

# =====================
# BACKTESTING ENGINE
# =====================
class BacktestEngine:
    def __init__(self):
        self.balance = initial_balance
        self.positions = {}
        self.trade_history = []
        self.current_position = None
        self.fee_per_share = 0.0035
        self.slippage = 0.0002
        self.trade_markers = []
        self.entry_conditions = {
            'ema_order': False,
            'rsi_threshold': False,
            'position_status': False
        }

    def calculate_fees(self, shares):
        return abs(shares) * self.fee_per_share

    def apply_slippage(self, price, is_buy=True):
        slippage = price * self.slippage
        return price + slippage if is_buy else price - slippage

    def execute_trade(self, ticker, price, shares, action, timestamp):
        executed_price = self.apply_slippage(price, action=='BUY')
        fees = self.calculate_fees(shares)
        position_value = abs(shares * executed_price)
        position_pct = (position_value / self.balance) * 100

        trade = {
            'timestamp': timestamp,
            'action': action,
            'price': executed_price,
            'shares': shares,
            'position_value': position_value,
            'position_pct': position_pct,
            'fees': fees,
            'new_balance': self.balance
        }

        if action == 'BUY':
            self.balance -= position_value + fees
            self.current_position = {
                'entry_price': executed_price,
                'shares': shares,
                'entry_time': timestamp,
                'fees_paid': fees
            }
        else:
            # Calculate PNL before updating balance
            if self.current_position:  # Added null check
                pnl = (executed_price - self.current_position['entry_price']) * shares
                pnl_pct = (pnl / self.current_position['entry_price']) * 100
                
                trade.update({
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                })
                
                self.balance += position_value - fees
                self.trade_history.append(trade.copy())
                self.current_position = None

    # Update marker configuration
        marker = {
            'date': timestamp,
            'price': executed_price,
            'marker': '↑' if action == 'BUY' else '↓',  # Using thin Unicode arrows
            'color': 'green' if action == 'BUY' else 'red',
            'size': 120  # Slightly larger size for better visibility
        }
        self.trade_markers.append(marker)
    
        return trade

# Initialize engine globally
engine = BacktestEngine()

# =====================
# STRATEGY CONDITIONS
# =====================
def check_entry_conditions(df, index):
    row = df.loc[index]
    conditions = {
        'ema_order': (row['1_EMA'] > row['3_EMA'] > row['8_EMA'] > row['VWAP']),
        'rsi_threshold': row['RSI'] < 70,
        'position_status': not engine.current_position
    }
    engine.entry_conditions = conditions
    return all(conditions.values())

def check_exit_conditions(df, index):
    if not engine.current_position:
        return False
        
    row = df.loc[index]
    exit_conditions = [
        row['1_EMA'] < row['3_EMA'],
        row['3_EMA'] < row['8_EMA'],
        row['RSI'] > 70,
        row['close'] < engine.current_position['entry_price'] * 0.98
    ]
    return any(exit_conditions)

# =====================
# ENHANCED LOGGING
# =====================
class TradeFormatter(logging.Formatter):
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    
    def format(self, record):
        if 'BUY' in record.msg:
            record.msg = f"{self.GREEN}{record.msg}{self.RESET}"
        elif 'SELL' in record.msg:
            record.msg = f"{self.RED}{record.msg}{self.RESET}"
        return super().format(record)

# Configure root logger directly
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Clear existing handlers
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Create and configure handlers
handler = logging.StreamHandler()
handler.setFormatter(TradeFormatter('%(asctime)s | %(message)s'))
root_logger.addHandler(handler)

file_handler = logging.FileHandler('strategy_execution.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)

def log_trade(trade, symbol, retries=3):
    """
    Enhanced trade logging with robust timestamp handling and YF data validation
    """
    # Validate trade structure first
    required_keys = {'timestamp', 'price', 'position_value', 'shares', 'action'}
    if missing := required_keys - trade.keys():
        logger.error(f"Missing trade keys: {missing}")
        return

    # Parse timestamp with proper timezone handling
    try:
        trade_time = pd.to_datetime(trade['timestamp'], utc=True)
        trade_time = trade_time.tz_convert('America/New_York')
        window_start = trade_time - pd.Timedelta(minutes=5)
        window_end = trade_time + pd.Timedelta(minutes=5)
    except Exception as e:
        logger.error(f"Invalid timestamp {trade['timestamp']}: {str(e)}")
        return

    # Yahoo Finance fetch with retries and headers
    yf_spot = None
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json'
    }
    
    for attempt in range(retries):
        try:
            data = yf.Ticker(symbol, headers=headers).history(
                start=window_start.strftime('%Y-%m-%d'),
                end=window_end.strftime('%Y-%m-%d'),
                interval='5m',
                prepost=True,
                timeout=10
            )
            
            if not data.empty:
                # Find nearest timestamp within 5 minutes
                idx = data.index.get_indexer([trade_time], method='nearest')[0]
                yf_spot = data.iloc[idx].Close
                break
                
        except Exception as e:
            if attempt < retries - 1:
                sleep(2 ** attempt)
            continue
    
    # Build log messages
    log_parts = [
        f"{trade['action']} | {symbol}",
        f"Time: {trade_time.strftime('%Y-%m-%d %H:%M:%S%z')}",
        f"Price: ${trade['price']:.2f}",
        f"YF Spot: ${yf_spot:.2f}" if yf_spot else "YF Spot: N/A",
        f"Size: ${trade['position_value']:.2f}",
        f"Shares: {trade['shares']:.2f}"
    ]
    
    if trade['action'] == 'SELL':
        log_parts.extend([
            f"PnL: ${trade['pnl']:.2f}",
            f"Return: {trade['pnl_pct']:.2f}%"
        ])
    
    logger.info(" | ".join(log_parts))

# =====================
# DATA MANAGEMENT
# =====================
nest_asyncio.apply()

def get_historical_data(ib, symbol, exchange='SMART', currency='USD', backtest=False):
    contract = Stock(symbol, exchange, currency)
    
    if backtest:
        # Modified duration and bar size settings
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',  # Empty string means current time
            durationStr='30 D',
            barSizeSetting='1 hour',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1,
            keepUpToDate=False  # Ensure we get historical data only
        )
        
        if not bars:
            raise ValueError("No data received from IB")
            
        df = util.df(bars)
        
        # Ensure we have data before calculating indicators
        if len(df) == 0:
            raise ValueError("Empty dataframe received from IB")
            
        # Calculate technical indicators
        for window in [1, 3, 8]:
            df[f'{window}_EMA'] = df['close'].ewm(span=window, adjust=False).mean()
        
        # Calculate VWAP properly for intraday data
        df['vwap_numerator'] = df['volume'] * (df['high'] + df['low'] + df['close']) / 3
        df['vwap_denominator'] = df['volume']
        
        # Group by date to reset VWAP calculations daily
        df['date'] = pd.to_datetime(df['date'])
        df['trading_date'] = df['date'].dt.date
        
        # Calculate VWAP for each trading day
        df['VWAP'] = (df.groupby('trading_date')['vwap_numerator'].cumsum() / 
                      df.groupby('trading_date')['vwap_denominator'].cumsum())
        
        # Clean up temporary columns
        df = df.drop(['vwap_numerator', 'vwap_denominator', 'trading_date'], axis=1)
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df.set_index('date', inplace=True)
        
        return df
    
    # Real-time data handling remains unchanged
    return get_multi_timeframe_data(ib, contract)

def get_multi_timeframe_data(ib, contract):
    timeframes = {
        '5m': ('5 mins', '1 D'),
        '30m': ('30 mins', '1 D'),
        '1h': ('1 hour', '1 D')
    }
    
    dfs = {}
    for tf, (bar_size, duration) in timeframes.items():
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=datetime.now(),
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        df = util.df(bars)
        
        # Add EMA calculations (CRITICAL FIX)
        df['close'] = df['close'].astype(float)
        for window in [1, 3, 8]:
            df[f'{window}_EMA'] = df['close'].ewm(span=window, adjust=False).mean()
        
        # Existing VWAP/volume calculations
        df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close'])/3).cumsum()/df['volume'].cumsum()
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df.set_index('date', inplace=True)
        dfs[tf] = df
        
    return dfs['5m'], dfs['30m'], dfs['1h']

# =====================
# TECHNICAL ANALYSIS
# ===================== 

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

# =====================
# VISUALIZATION ENGINE
# =====================
def plot_candlestick(df, backtest=False):
    addplots = []
    
    # Add technical indicators using EMA dictionary mapping
    ema_colors = {1: 'blue', 3: 'orange',8: 'red'}
    for ema in [1, 3, 8]:
        addplots.append(mpf.make_addplot(
            df[f'{ema}_EMA'],
            color=ema_colors[ema],
            width=1.5,
            alpha=0.8
        ))
    
    addplots.append(mpf.make_addplot(
        df['VWAP'],
        color='purple',
        width=1.5,
        alpha=0.8
    ))
    
    # Handle trade markers for backtest mode
    if backtest and engine.trade_markers:
        markers_df = pd.DataFrame(engine.trade_markers)
        marker_series = pd.Series(np.nan, index=df.index)
        
        # Fill marker series with prices at correct dates
        for _, row in markers_df.iterrows():
            marker_series.at[row['date']] = row['price']
        
        # Plot buy markers
        buy_mask = markers_df['marker'] == '↑'
        if any(buy_mask):
            buy_series = marker_series.copy()
            buy_series[~marker_series.index.isin(markers_df[buy_mask]['date'])] = np.nan
            addplots.append(mpf.make_addplot(
                buy_series,
                type='scatter',
                markersize=80,
                marker='$\\uparrow$',  # Thin up arrow
                color='#00FF00'  # Bright green
            ))
        
        # Plot sell markers
        sell_mask = markers_df['marker'] == '↓'
        if any(sell_mask):
            sell_series = marker_series.copy()
            sell_series[~marker_series.index.isin(markers_df[sell_mask]['date'])] = np.nan
            addplots.append(mpf.make_addplot(
                sell_series,
                type='scatter',
                markersize=80,
                marker='$\\downarrow$',  # Thin down arrow
                color='#FF0000'  # Bright red
            ))
    
    # Plot configuration
    mc = mpf.make_marketcolors(
        up='green',
        down='red',
        edge='inherit',
        wick='inherit',
        volume={'up':'green', 'down':'red'}
    )
    
    s = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle='dotted',
        base_mpf_style='charles'
    )
    
    # Create the plot
    fig, axlist = mpf.plot(
        df,
        type='candle',
        style=s,
        volume=True,
        addplot=addplots,
        figsize=(12, 8),
        panel_ratios=(6, 2),
        returnfig=True
    )
    
    # Add legend
    ax = axlist[0]
    ax.legend(
        handles=[
            Patch(facecolor="blue", label="1 EMA"),
            Patch(facecolor="orange", label="3 EMA"),
            Patch(facecolor="red", label="8 EMA"),
            Patch(facecolor="purple", label="VWAP")
        ],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.1),
        ncol=4
    )
    
    plt.show()

# =====================
# EXECUTION HANDLERS (Critical Fix)
# =====================
def run_backtest(df):
    """Fixed index handling with proper data validation"""
    # Validate dataframe structure and index
    required_columns = ['open', 'high', 'low', 'close', '1_EMA', '3_EMA', '8_EMA', 'RSI']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")

    # Convert index to datetime and sort
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index(drop=False)
        if 'date' not in df.columns:
            raise KeyError("DataFrame must contain 'date' column for index conversion")
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
    # Fixed loop structure
    for i, current_dt in enumerate(df.index):
        try:
            row = df.loc[current_dt]
            
            # ENTRY CONDITION CHECK
            if check_entry_conditions(df, current_dt):
                position_size = min(
                    engine.balance * 0.02,
                    engine.balance - 100,
                    key=lambda x: x if x > 0 else 0
                )
                
                if row['close'] <= 0 or pd.isna(row['close']):
                    logger.error(f"Skipping invalid price at {current_dt}")
                    continue
                    
                shares = max(0.01, position_size / row['close'])
                
                # Execute buy order
                trade = engine.execute_trade(symbol, row['close'], shares, 'BUY', current_dt)
                log_trade(trade, symbol)
            
            # EXIT CONDITION CHECK
            if engine.current_position and check_exit_conditions(df, current_dt):
                if any(pd.isna(row[col]) for col in ['1_EMA', '3_EMA', '8_EMA', 'RSI', 'close']):
                    logger.error(f"Skipping exit due to missing data at {current_dt}")
                    continue
                    
                # Execute sell order
                trade = engine.execute_trade(symbol, row['close'], engine.current_position['shares'], 'SELL', current_dt)
                log_trade(trade, symbol)
                
        except KeyError as ke:
            logger.error(f"Data access error: {str(ke)}")
            break
        except Exception as e:
            logger.error(f"Error processing {current_dt}: {str(e)}")
            raise

    # Post-backtest cleanup
    if engine.current_position:
        logger.warning(f"Exiting remaining position at market close")
        last_price = df.iloc[-1]['close']
        trade = engine.execute_trade(
            symbol, 
            last_price, 
            engine.current_position['shares'], 
            'SELL', 
            df.index[-1]
        )

# =====================
# MAIN EXECUTION
# =====================
def main():
    ib = IB()
    ib.connect('127.0.0.1', 7497, clientId=1)

    try:
        # Backtest mode
        daily_data = get_historical_data(ib, symbol, backtest=True)
        run_backtest(daily_data)
        plot_candlestick(daily_data, backtest=True)
        
        # Real-time mode
        df_5m, df_30m, df_1h = get_historical_data(ib, symbol)
        plot_candlestick(df_5m)
        
    except Exception as e:
        logger.error(f"Execution error: {str(e)}")
    finally:
        if 'ib' in locals() and ib.isConnected():
            ib.disconnect()
            
            if engine.trade_history:
                trade_history = pd.DataFrame(engine.trade_history)
                
                # Calculate performance metrics
                total_trades = len(trade_history)
                profitable_trades = (trade_history['pnl'] > 0).sum()
                losing_trades = (trade_history['pnl'] < 0).sum()
                win_rate = (trade_history['pnl'] > 0).mean() * 100
                
                # Calculate returns and risk metrics
                total_pnl = trade_history['pnl'].sum()
                returns = trade_history['pnl'] / initial_balance
                annualized_return = ((engine.balance/initial_balance) ** (252/len(returns)) - 1) * 100
                volatility = returns.std() * np.sqrt(252)
                
                # Calculate Sharpe and Sortino ratios
                risk_free_rate = 0.02  # 2% annual risk-free rate
                excess_returns = returns - (risk_free_rate/252)
                sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std()
                downside_returns = returns[returns < 0]
                sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std()
                
                # Calculate drawdown
                cumulative_returns = (1 + returns).cumprod()
                rolling_max = cumulative_returns.expanding().max()
                drawdowns = (cumulative_returns - rolling_max) / rolling_max
                max_drawdown = drawdowns.min() * 100
                
                # Calculate trade statistics
                avg_profit = trade_history[trade_history['pnl'] > 0]['pnl'].mean()
                avg_loss = abs(trade_history[trade_history['pnl'] < 0]['pnl'].mean())
                profit_factor = abs(trade_history[trade_history['pnl'] > 0]['pnl'].sum() / 
                                trade_history[trade_history['pnl'] < 0]['pnl'].sum())
                
                logger.info("\n=== Backtest Summary ===")
                logger.info("Initial Balance: $%.2f", initial_balance)
                logger.info("Final Balance: $%.2f", engine.balance)
                logger.info("Total Return: %.2f%%", ((engine.balance/initial_balance - 1) * 100))
                logger.info("Annualized Return: %.2f%%", annualized_return)
                logger.info("\n=== Trade Statistics ===")
                logger.info("Total Trades: %d", total_trades)
                logger.info("Profitable Trades: %d", profitable_trades)
                logger.info("Losing Trades: %d", losing_trades)
                logger.info("Win Rate: %.2f%%", win_rate)
                logger.info("Average Win: $%.2f", avg_profit)
                logger.info("Average Loss: $%.2f", avg_loss)
                logger.info("Profit Factor: %.2f", profit_factor)
                logger.info("\n=== Risk Metrics ===")
                logger.info("Sharpe Ratio: %.2f", sharpe_ratio)
                logger.info("Sortino Ratio: %.2f", sortino_ratio)
                logger.info("Maximum Drawdown: %.2f%%", max_drawdown)
                logger.info("Annualized Volatility: %.2f%%", volatility * 100)
                logger.info("======================")

if __name__ == "__main__":
    main()