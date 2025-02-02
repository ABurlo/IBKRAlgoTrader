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

# =====================
# GLOBAL CONFIGURATION
# =====================
symbol = "MSFT"  # Default Ticker
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
            pnl = (executed_price - self.current_position['entry_price']) * shares
            pnl_pct = (pnl / self.current_position['entry_price']) * 100
            
            trade.update({
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })
            
            self.balance += position_value - fees
            self.trade_history.append(trade.copy())
            self.current_position = None

        # Add visual marker
        marker = {
            'date': timestamp,
            'price': executed_price,
            'marker': '▼' if action == 'SELL' else '▲',
            'color': 'red' if action == 'SELL' else 'green',
            'size': 100
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
        'ema_order': (row['5_EMA'] > row['10_EMA'] > row['20_EMA'] > row['VWAP']),
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
        row['5_EMA'] < row['10_EMA'],
        row['10_EMA'] < row['20_EMA'],
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

# =====================
# DATA MANAGEMENT
# =====================
nest_asyncio.apply()

def get_historical_data(ib, symbol, exchange='SMART', currency='USD', backtest=False):
    contract = Stock(symbol, exchange, currency)
    
    if backtest:
        # Backtesting data (1 year daily)
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=datetime.now(),
            durationStr='365 D',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        df = util.df(bars)
        
        # Calculate technical indicators
        for window in [5, 10, 20]:
            df[f'{window}_EMA'] = df['close'].ewm(span=window, adjust=False).mean()
        
        df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
            # Add index validation
        df = df[~df.index.duplicated(keep='first')]
        df = df.asfreq('D').ffill()
        
        # Ensure continuous index
        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        df = df.reindex(full_index).ffill().bfill()
        
        return df
    
    # Real-time data (multi-timeframe)
    return get_multi_timeframe_data(ib, contract)

def get_multi_timeframe_data(ib, contract):
    # Original multi-timeframe implementation
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
        
        # Calculate VWAP and volume metrics
        df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['relative_volume'] = df['volume'] / df['volume_ma']
        
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
    
    # Add technical indicators
    for ema in [5, 10, 20]:
        addplots.append(mpf.make_addplot(
            df[f'{ema}_EMA'],
            color=['blue', 'orange', 'red'][ema//5-1],
            width=1.5,
            alpha=0.8
        ))
    
    addplots.append(mpf.make_addplot(
        df['VWAP'],
        color='purple',
        width=1.5,
        alpha=0.8
    ))

    # Add trade markers
    if backtest and engine.trade_markers:
        markers = pd.DataFrame(engine.trade_markers)
        addplots.append(mpf.make_addplot(
            markers.set_index('date')['price'],
            type='scatter',
            marker=markers['marker'].values,
            color=markers['color'].values,
            markersize=100
        ))

    # Plot configuration
    mc = mpf.make_marketcolors(
        up='green', down='red',
        edge='inherit', wick='inherit',
        volume={'up':'green', 'down':'red'}
    )
    
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='dotted')
    
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
    
    # Configure axes and legends
    ax = axlist[0]
    ax.legend(
        handles=[
            Patch(facecolor="blue", label="5 EMA"),
            Patch(facecolor="orange", label="10 EMA"),
            Patch(facecolor="red", label="20 EMA"),
            Patch(facecolor="purple", label="VWAP")
        ],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.1),
        ncol=4
    )
    
    plt.show()

# =====================
# EXECUTION HANDLERS
# =====================
def run_backtest(df):
    """Execute backtest with robust index handling and trade safeguards"""
    # Validate dataframe structure and index
    required_columns = ['open', 'high', 'low', 'close', '5_EMA', '10_EMA', '20_EMA', 'RSI']
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
        
    # Pre-calculate valid indices
    valid_indices = df.index.tolist()
    
    for i, current_dt in enumerate(valid_indices):
        try:
            row = df.loc[current_dt]
            
            # ====================
            # ENTRY CONDITION CHECK
            # ====================
            if check_entry_conditions(df, current_dt):
                # Position sizing safeguards
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
                
                logger.info(
                    f"BUY | {symbol} | Size: ${trade['position_value']:.2f} "
                    f"| Shares: {trade['shares']:.2f} | Balance%: {trade['position_pct']:.2f}% "
                    f"| Price: ${trade['price']:.2f} | New Balance: ${trade['new_balance']:.2f}"
                )

            # ===================
            # EXIT CONDITION CHECK
            # ===================
            if engine.current_position and check_exit_conditions(df, current_dt):
                # Validate technical indicators
                if any(pd.isna(row[col]) for col in ['5_EMA', '10_EMA', 'RSI', 'close']):
                    logger.error(f"Skipping exit due to missing data at {current_dt}")
                    continue
                    
                # Execute sell order
                trade = engine.execute_trade(
                    symbol, 
                    row['close'], 
                    engine.current_position['shares'], 
                    'SELL', 
                    current_dt
                )
                
                logger.info(
                    f"SELL | {symbol} | PnL: ${trade['pnl']:.2f} "
                    f"| Return: {trade['pnl_pct']:.2f}% | Price: ${trade['price']:.2f} "
                    f"| New Balance: ${trade['new_balance']:.2f}"
                )

        except KeyError as ke:
            logger.error(f"Data access error: {str(ke)}")
            break
        except IndexError as ie:
            logger.error(f"Index mismatch at position {i}/{len(df)}: {str(ie)}")
            break
        except ZeroDivisionError:
            logger.error(f"Zero price encountered at {current_dt}")
            break
        except Exception as e:
            logger.error(f"Critical error at {current_dt}: {str(e)}")
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
        ib.disconnect()
        logger.info("Final Balance: $%.2f | Total Trades: %d", 
                   engine.balance, len(engine.trade_history))

if __name__ == "__main__":
    main()
