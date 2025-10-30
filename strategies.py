# ==========================================================
# ADVANCED MULTI-STRATEGY BACKTEST WITH FUNDAMENTAL FEATURES
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Load datasets
# -----------------------------
prices = pd.read_csv("/mnt/data/prices-split-adjusted.csv")
securities = pd.read_csv("/mnt/data/securities.csv")
fundamentals = pd.read_csv("/mnt/data/fundamentals.csv")

# Clean & prepare
prices['date'] = pd.to_datetime(prices['date'])
prices = prices.sort_values(['symbol', 'date'])
prices = prices[['date', 'symbol', 'close', 'volume']].rename(columns={'close': 'adj_close'})

# -----------------------------
# 2. Data filtering and pivot
# -----------------------------
avg_vol = prices.groupby('symbol')['volume'].mean()
top_symbols = avg_vol.nlargest(200).index.tolist()
prices = prices[prices['symbol'].isin(top_symbols)]

price_pivot = prices.pivot(index='date', columns='symbol', values='adj_close').sort_index()

# -----------------------------
# 3. Prepare fundamental indicators
# -----------------------------
fundamentals.rename(columns={'Ticker Symbol': 'symbol'}, inplace=True)
fundamentals = fundamentals.dropna(subset=['Period Ending'])
fundamentals['Period Ending'] = pd.to_datetime(fundamentals['Period Ending'])
fundamentals = fundamentals.sort_values(['symbol', 'Period Ending'])

# Compute rolling fundamentals per ticker
def compute_fundamentals(df):
    df = df.copy()
    df['ROE'] = df['After Tax ROE']
    df['CashRatio'] = df['Cash Ratio']
    df['CapEx'] = df['Capital Expenditures']
    df['Receivable_to_Payable'] = df['Accounts Receivable'] / (df['Accounts Payable'] + 1e-9)
    df['CapitalSurplus'] = df['Capital Surplus']
    return df[['symbol', 'Period Ending', 'ROE', 'CashRatio', 'CapEx', 'Receivable_to_Payable', 'CapitalSurplus']]

fundamentals = compute_fundamentals(fundamentals)

# Map latest fundamentals to each date for ML
latest_fundamentals = fundamentals.groupby('symbol').apply(lambda x: x.ffill().iloc[-1]).reset_index(drop=True)

# -----------------------------
# 4. Helper functions
# -----------------------------
def portfolio_stats(portf_returns, periods_per_year=252):
    cumret = (1 + portf_returns).cumprod().iloc[-1] - 1
    ann_ret = (1 + portf_returns.mean()) ** periods_per_year - 1
    ann_vol = portf_returns.std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum = (1 + portf_returns).cumprod()
    dd = cum.cummax() - cum
    mdd = (dd / cum.cummax()).max()
    return {
        'Cumulative Return': cumret,
        'Annual Return': ann_ret,
        'Annual Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': mdd
    }

# -----------------------------
# 5. Trading strategies
# -----------------------------
def long_only_momentum(price_df, lookback=90, top_n=30, rebalance_freq='M', cost=0.0005):
    returns = price_df.pct_change()
    rebalance_dates = price_df.resample(rebalance_freq).last().index
    weights = pd.DataFrame(0, index=price_df.index, columns=price_df.columns)
    for i in range(lookback, len(rebalance_dates)):
        end = rebalance_dates[i]
        start = rebalance_dates[i - 1]
        look_start = end - pd.Timedelta(days=lookback)
        if look_start < price_df.index[0]:
            continue
        past_returns = price_df.loc[end] / price_df.loc[look_start] - 1
        past_returns = past_returns.dropna()
        selected = past_returns.nlargest(top_n).index
        weights.loc[(price_df.index > start) & (price_df.index <= end), selected] = 1 / top_n
    port_ret = (weights.shift(1) * returns).sum(axis=1)
    turnover = (weights - weights.shift(1).fillna(0)).abs().sum(axis=1)
    port_ret -= turnover * cost
    return port_ret


def long_short_momentum(price_df, lookback=90, top_n=30, rebalance_freq='M', cost=0.0005):
    returns = price_df.pct_change()
    rebalance_dates = price_df.resample(rebalance_freq).last().index
    weights = pd.DataFrame(0, index=price_df.index, columns=price_df.columns)
    for i in range(lookback, len(rebalance_dates)):
        end = rebalance_dates[i]
        start = rebalance_dates[i - 1]
        look_start = end - pd.Timedelta(days=lookback)
        if look_start < price_df.index[0]:
            continue
        past_returns = price_df.loc[end] / price_df.loc[look_start] - 1
        past_returns = past_returns.dropna().sort_values()
        longs = past_returns.tail(top_n).index
        shorts = past_returns.head(top_n).index
        weights.loc[(price_df.index > start) & (price_df.index <= end), longs] = 0.5 / top_n
        weights.loc[(price_df.index > start) & (price_df.index <= end), shorts] = -0.5 / top_n
    port_ret = (weights.shift(1) * returns).sum(axis=1)
    turnover = (weights - weights.shift(1).fillna(0)).abs().sum(axis=1)
    port_ret -= turnover * cost
    return port_ret


def ma_crossover(price_df, short=50, long=200, cost=0.0005):
    sma_short = price_df.rolling(short).mean()
    sma_long = price_df.rolling(long).mean()
    signal = (sma_short > sma_long).astype(int)
    weights = signal.div(signal.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    ret = price_df.pct_change()
    port_ret = (weights.shift(1) * ret).sum(axis=1)
    turnover = (weights - weights.shift(1).fillna(0)).abs().sum(axis=1)
    port_ret -= turnover * cost
    return port_ret


# --- 5.4 Machine Learning Strategy (Enhanced with Fundamentals) ---
def ml_hybrid_strategy(price_df, fundamentals_df, top_n=20, cost=0.0005):
    weekly_price = price_df.resample('W-FRI').last().dropna(how='all')
    weekly_ret = weekly_price.pct_change().fillna(0)
    rets = pd.Series(0.0, index=weekly_price.index)
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

    # Standardize fundamentals
    f_scaled = fundamentals_df.copy()
    f_scaled.iloc[:, 2:] = StandardScaler().fit_transform(f_scaled.iloc[:, 2:])

    for i in range(52, len(weekly_price) - 1, 4):  # каждые 4 недели
        hist = weekly_price.iloc[:i]
        future = weekly_price.iloc[i + 1]
        r4 = hist.pct_change(4).iloc[-1]
        r12 = hist.pct_change(12).iloc[-1]
        vol12 = hist.pct_change().rolling(12).std().iloc[-1]
        df = pd.DataFrame({'r4': r4, 'r12': r12, 'vol12': vol12}).dropna()

        # merge with fundamentals
        df = df.merge(f_scaled, left_index=True, right_on='symbol', how='left').set_index('symbol')
        df = df.fillna(df.median())

        next_ret = (future / hist.iloc[-1] - 1).dropna()
        label = (next_ret > 0).astype(int).reindex(df.index)

        model.fit(df, label)
        probs = model.predict_proba(df)[:, 1]
        selected = df.index[np.argsort(probs)[-top_n:]]
        week_ret = next_ret[selected].mean()
        rets.iloc[i + 1] = week_ret

    rets = rets.reindex(price_df.index, method='ffill').fillna(0)
    return rets


# -----------------------------
# 6. Run backtests
# -----------------------------
lom_ret = long_only_momentum(price_pivot)
lsm_ret = long_short_momentum(price_pivot)
mac_ret = ma_crossover(price_pivot)
ml_ret = ml_hybrid_strategy(price_pivot, latest_fundamentals)

# -----------------------------
# 7. Compare performance
# -----------------------------
strategies = {
    'Long-Only Momentum': lom_ret,
    'Long-Short Momentum': lsm_ret,
    'MA Crossover': mac_ret,
    'Hybrid ML (Tech + Fundamentals)': ml_ret
}

results = {name: portfolio_stats(ret) for name, ret in strategies.items()}
results_df = pd.DataFrame(results).T.sort_values('Sharpe Ratio', ascending=False)
print(results_df)

# -----------------------------
# 8. Plot cumulative returns
# -----------------------------
plt.figure(figsize=(10,6))
for name, ret in strategies.items():
    plt.plot((1 + ret).cumprod(), label=name)
plt.legend()
plt.title("Cumulative Returns of Trading Strategies")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.show()

results_df.to_csv("/mnt/data/strategy_results_with_fundamentals.csv")
print("✅ Results saved to /mnt/data/strategy_results_with_fundamentals.csv")
