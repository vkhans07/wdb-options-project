"""
generate_validation_charts.py
═══════════════════════════════════════════════════════════════════════════════
Standalone script — generates ALL charts for the three real-world bidding
war validation cases. Run this independently of merger_analysis.py.

For each case you get 3 charts:
  1. Normalized returns (Base 100) across the full study window with
     all deal milestones marked — the "Winner's Curse" visual
  2. Spread Z-Score — the beta-adjusted divergence signal with ±2σ bands
     and entry date marked — the "here's when we enter" visual
  3. 3-panel backtest PnL / legs comparison / drawdown chart

Cases:
  ① Anadarko (2019) — CVX vs OXY   ← cleanest bidding war analog
  ② Hulu (2023)     — CMCSA vs DIS
  ③ KSU (2021)      — CNI vs CP

Hedge ratio methodology: CONSTANT beta estimated via OLS on the
pre-entry window only. Locked in at trade entry, never rebalanced.

Requirements:
  pip install yfinance statsmodels pandas numpy matplotlib scipy seaborn

Output: all charts saved to charts/validation/
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.transforms as transforms
import seaborn as sns
import os


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit dates, tickers, and milestones here only
# ══════════════════════════════════════════════════════════════════════════════

CASES = {
    'anadarko': {
        'title':       'Anadarko Bidding War (2019): Chevron vs Occidental',
        'long_ticker':  'CVX',
        'short_ticker': 'OXY',
        'long_label':   'CVX (walked away, $1B fee)',
        'short_label':  'OXY (won, $38B debt burden)',
        'fetch_start':  '2019-02-01',   # ~3 months before first bid
        'fetch_end':    '2019-12-31',
        'entry_date':   '2019-05-05',   # Chevron officially withdraws
        'milestones': {
            '2019-04-12': 'Chevron $33B bid',
            '2019-04-24': 'OXY $38B counterbid',
            '2019-05-05': 'Chevron walks ← ENTRY',
            '2019-08-08': 'Deal closes',
        },
    },
    'hulu': {
        'title':       'Hulu Buyout (2023): Disney vs Comcast',
        'long_ticker':  'CMCSA',
        'short_ticker': 'DIS',
        'long_label':   'CMCSA (walked away with cash)',
        'short_label':  'DIS (absorbed ~$8.6B Hulu liability)',
        'fetch_start':  '2023-08-01',
        'fetch_end':    '2024-06-30',
        'entry_date':   '2023-11-01',   # Disney/Comcast reach valuation agreement
        'milestones': {
            '2023-09-06': 'Disney invokes put option',
            '2023-11-01': 'Valuation agreed ← ENTRY',
            '2024-02-01': 'Disney full ownership complete',
        },
    },
    'ksu': {
        'title':       'KSU Railway War (2021): Canadian Pacific vs Canadian National',
        'long_ticker':  'CNI',
        'short_ticker': 'CP',
        'long_label':   'CNI (walked away, $700M breakup fee)',
        'short_label':  'CP (won KSU, took on heavy debt)',
        'fetch_start':  '2021-05-01',   # warm-up before CNI counterbid
        'fetch_end':    '2022-09-15',   # 1 year post-entry
        'entry_date':   '2021-09-15',   # STB forces CN out; CP declared winner
        'milestones': {
            '2021-03-21': 'CP initial $25B bid',
            '2021-05-13': 'CNI $33.7B counterbid',
            '2021-09-15': 'STB blocks CNI ← ENTRY',
            '2022-04-14': 'CP-KSU merger closes',
        },
    },
}

# Backtest parameters — identical to merger_analysis.py
CAPITAL      = 100_000
RISK_FREE    = 0.045
TRADING_DAYS = 252
BORROW_COST  = 0.0050
HEDGE_WINDOW = 20
OUTPUT_DIR   = 'charts/validation'


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def setup_style():
    sns.set_theme(style='whitegrid', context='talk')
    plt.rcParams.update({
        'figure.dpi':       300,
        'savefig.dpi':      300,
        'savefig.bbox':     'tight',
        'font.family':      'sans-serif',
        'axes.titleweight': 'bold',
    })


def format_x_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def add_case_milestones(ax, milestones):
    """Generic milestone annotator — takes a dict of {date_str: label}."""
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for date_str, label in milestones.items():
        d = pd.to_datetime(date_str)
        ax.axvline(d, color='black', linestyle='-.', alpha=0.6, linewidth=1.5)
        ax.text(d, 0.95, f' {label}', transform=trans,
                rotation=90, va='top', ha='right',
                fontsize=10, fontweight='bold', color='#333333')


def fetch_data(long_ticker, short_ticker, start, end):
    """Download adjusted closes, return clean DataFrame with LONG / SHORT cols."""
    print(f"  Downloading {long_ticker} & {short_ticker} ({start} → {end})...")
    raw   = yf.download([long_ticker, short_ticker],
                        start=start, end=end,
                        auto_adjust=True, progress=False)
    close = raw['Close'][[long_ticker, short_ticker]].copy()
    close.columns = ['LONG', 'SHORT']
    close = close.reset_index().rename(columns={'index': 'Date', 'Datetime': 'Date'})
    close['Date'] = pd.to_datetime(close['Date']).dt.tz_localize(None)
    close = close.sort_values('Date').reset_index(drop=True).ffill().dropna()
    print(f"  {len(close)} trading days loaded.")
    return close


def compute_hedge_ratio(df, window=HEDGE_WINDOW):
    """Rolling beta of LONG vs SHORT, lagged 1 day."""
    lr       = df['LONG'].pct_change()
    sr       = df['SHORT'].pct_change()
    beta     = lr.rolling(window).cov(sr) / sr.rolling(window).var()
    return beta.shift(1).fillna(1.0)


def compute_spread_zscore(df, hedge_ratio, window=HEDGE_WINDOW):
    """Beta-adjusted log-return spread, normalised to a z-score."""
    lr              = np.log(df['LONG']  / df['LONG'].shift(1))
    sr              = np.log(df['SHORT'] / df['SHORT'].shift(1))
    spread          = lr - hedge_ratio * sr
    mean            = spread.rolling(window).mean()
    std             = spread.rolling(window).std()
    zscore          = (spread - mean) / std
    return spread, zscore


def run_backtest(df, entry_date):
    """
    Beta-neutral pairs backtest using a CONSTANT hedge ratio.

    Beta is estimated once via OLS on the pre-entry window only:
        β = Cov(LONG_ret, SHORT_ret) / Var(SHORT_ret)
    This single value is locked in at trade entry and held fixed for
    the entire trade window — no rebalancing, no look-ahead bias.

    Using a constant beta is correct here because:
      - We enter on a single catalyst date and hold; there is no
        ongoing rebalancing logic in this strategy
      - Rolling beta would implicitly resize the position daily,
        adding unmodelled transaction costs
      - The pre-entry window gives the cleanest estimate of the
        structural relationship before deal noise contaminates it
    """
    entry    = pd.to_datetime(entry_date)
    df       = df.copy()
    pre      = df[df['Date'] < entry].copy()

    if len(pre) < HEDGE_WINDOW:
        print(f"  ⚠  Only {len(pre)} pre-entry days — beta estimate unreliable.")

    # OLS beta on pre-entry returns
    pre_long_ret  = pre['LONG'].pct_change().dropna()
    pre_short_ret = pre['SHORT'].pct_change().dropna()
    aligned       = pd.concat([pre_long_ret, pre_short_ret], axis=1).dropna()
    aligned.columns = ['L', 'S']

    if len(aligned) >= 2:
        const_beta = aligned['L'].cov(aligned['S']) / aligned['S'].var()
    else:
        const_beta = 1.0
        print("  ⚠  Insufficient pre-entry data — defaulting β = 1.0")

    tw = df[df['Date'] >= entry].copy().reset_index(drop=True)
    if tw.empty:
        return None, None

    tw['LONG_Ret']  = tw['LONG'].pct_change().fillna(0)
    tw['SHORT_Ret'] = tw['SHORT'].pct_change().fillna(0)
    daily_borrow    = BORROW_COST / TRADING_DAYS

    tw['LONG_Daily_PnL']  = CAPITAL * tw['LONG_Ret']
    tw['SHORT_Daily_PnL'] = (-CAPITAL * tw['SHORT_Ret']
                             - CAPITAL * daily_borrow)
    tw['PAIR_Daily_PnL']  = (CAPITAL * tw['LONG_Ret']
                             - CAPITAL * const_beta * tw['SHORT_Ret']
                             - CAPITAL * const_beta * daily_borrow)

    # Cumulative return as % of capital
    for leg in ('LONG', 'SHORT', 'PAIR'):
        tw[f'{leg}_Cum_Ret'] = (tw[f'{leg}_Daily_PnL'].cumsum() / CAPITAL) * 100

    # Drawdown as % of capital
    tw['PAIR_Peak_Ret']     = tw['PAIR_Cum_Ret'].cummax()
    tw['PAIR_Drawdown_Pct'] = tw['PAIR_Cum_Ret'] - tw['PAIR_Peak_Ret']

    def _sharpe(pnl, scale):
        s = pnl.iloc[1:]
        e = (s / CAPITAL) - (RISK_FREE / TRADING_DAYS)
        return (e.mean() / e.std()) * np.sqrt(scale) if e.std() != 0 else np.nan

    metrics = {
        'Final Ret %':      tw['PAIR_Cum_Ret'].iloc[-1],
        'Max Drawdown %':   (tw['PAIR_Cum_Ret'] - tw['PAIR_Cum_Ret'].cummax()).min(),
        'Sharpe (Period)':  _sharpe(tw['PAIR_Daily_PnL'], len(tw) - 1),
        'Sharpe (Annual)':  _sharpe(tw['PAIR_Daily_PnL'], TRADING_DAYS),
        'Win Rate':         (tw['PAIR_Daily_PnL'] > 0).mean() * 100,
        'Const Beta':       const_beta,
        'Pre-Entry Days':   len(aligned),
    }

    return tw, metrics


# ══════════════════════════════════════════════════════════════════════════════
# CHART GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def chart_normalized_returns(df, case, case_key):
    """
    Chart 1 of 3: Normalized price returns (Base 100) across the full
    study window. Both tickers start at 100 on the first date of the
    fetch window so relative performance is immediately readable.
    Milestones are marked. Entry date shaded.
    """
    fig, ax = plt.subplots(figsize=(16, 8))

    entry = pd.to_datetime(case['entry_date'])

    colors = {'LONG': '#1f77b4', 'SHORT': '#d62728'}
    for col, label, color in [
        ('LONG',  case['long_label'],  colors['LONG']),
        ('SHORT', case['short_label'], colors['SHORT']),
    ]:
        norm = (df[col] / df[col].iloc[0]) * 100
        ax.plot(df['Date'], norm, label=label, color=color, linewidth=2.5)

    # Shade post-entry region
    ax.axvspan(entry, df['Date'].max(), color='#2ca02c', alpha=0.05,
               label='Trade Active')
    ax.axhline(100, color='black', linestyle=':', alpha=0.4, linewidth=1)

    add_case_milestones(ax, case['milestones'])

    ax.set_title(f"{case['title']}\nNormalized Returns (Base 100) — Full Study Window",
                 pad=20)
    ax.set_ylabel('Normalized Price (Base 100)')
    ax.legend(loc='lower left', fontsize=11)
    format_x_axis(ax)
    plt.tight_layout()

    path = f"{OUTPUT_DIR}/{case_key}_1_normalized_returns.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [1/3] Saved normalized returns → {path}")


def chart_spread_zscore(df, case, case_key):
    """
    Chart 2 of 3: Beta-adjusted spread z-score across the full study
    window. ±1σ and ±2σ reference bands. Entry date marked with a
    vertical line. Peak post-entry divergence annotated.

    This is the quantitative entry-signal slide — the spread breaking
    out of ±2σ on the entry date is the trade trigger.
    """
    hedge = compute_hedge_ratio(df)
    _, zscore = compute_spread_zscore(df, hedge)

    fig, ax = plt.subplots(figsize=(16, 8))

    entry = pd.to_datetime(case['entry_date'])

    ax.plot(df['Date'], zscore, color='#1f77b4', linewidth=2,
            label='Spread Z-Score (β-Adjusted)', zorder=3)

    # Reference bands
    for level, style, lbl in [(1, ':', '±1σ'), (2, '--', '±2σ Entry Band')]:
        ax.axhline( level, color='gray', linestyle=style, alpha=0.7,
                    linewidth=1.2, label=lbl if level == 2 else None)
        ax.axhline(-level, color='gray', linestyle=style, alpha=0.7,
                    linewidth=1.2)

    ax.axhline(0, color='black', linestyle='-', alpha=0.25, linewidth=1)

    # Shade trade-active region
    ax.axvspan(entry, df['Date'].max(), color='#2ca02c', alpha=0.06,
               label='Trade Active (Post Entry)')

    # Entry marker
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.axvline(entry, color='black', linestyle='-.', linewidth=1.5, alpha=0.8)
    ax.text(entry, 0.95, '  Trade Entry', transform=trans,
            rotation=90, va='top', ha='right',
            fontsize=11, fontweight='bold', color='#333333')

    # Annotate peak post-entry divergence
    post   = zscore[df['Date'] >= entry].dropna()
    if not post.empty:
        peak_val  = post.abs().max()
        peak_idx  = post.abs().idxmax()
        peak_date = df.loc[peak_idx, 'Date']
        peak_z    = zscore.loc[peak_idx]
        ax.annotate(
            f'Peak divergence\nZ = {peak_z:.2f}σ',
            xy=(peak_date, peak_z),
            xytext=(peak_date, peak_z + (1.0 if peak_z > 0 else -1.0)),
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
            fontsize=10, color='#d62728', fontweight='bold', ha='center',
        )

    ax.set_title(
        f"{case['title']}\n"
        f"Pairs Spread Z-Score — β-Adjusted LONG log-return minus β × SHORT log-return\n"
        f"(20-day rolling normalisation)",
        pad=20
    )
    ax.set_ylabel('Z-Score (σ)')
    ax.legend(loc='upper left', fontsize=10)
    format_x_axis(ax)
    plt.tight_layout()

    path = f"{OUTPUT_DIR}/{case_key}_2_spread_zscore.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [2/3] Saved spread z-score     → {path}")


def chart_backtest_pnl(df, tw, metrics, case, case_key):
    """
    Chart 3 of 3: 3-panel backtest chart.
      Panel 1 — Pairs strategy cumulative return (%) with profit/loss shading
      Panel 2 — All three legs overlaid (long only, short only, pairs)
      Panel 3 — Drawdown (%)
    Constant beta is reported in the Panel 1 title.
    """
    if tw is None or metrics is None:
        print(f"  [3/3] Skipped — backtest returned no data.")
        return

    entry      = pd.to_datetime(case['entry_date'])
    dates      = tw['Date']
    pair_final = metrics['Final Ret %']
    pair_sp    = metrics['Sharpe (Period)']
    pair_mdd   = metrics['Max Drawdown %']
    const_beta = metrics['Const Beta']

    long_final   = tw['LONG_Cum_Ret'].iloc[-1]
    short_final  = tw['SHORT_Cum_Ret'].iloc[-1]
    long_ticker  = case['long_ticker']
    short_ticker = case['short_ticker']

    pct_fmt = plt.FuncFormatter(lambda v, _: f'{v:.1f}%')

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 14),
        gridspec_kw={'height_ratios': [3, 2, 1.5]},
        sharex=True
    )

    # ── Panel 1: Pairs cumulative return ──────────────────────────────
    ax1 = axes[0]
    ax1.plot(dates, tw['PAIR_Cum_Ret'], color='#2ca02c', linewidth=2.5,
             label=f"Long {case['long_label']} / Short {case['short_label']} (β-Neutral)")
    ax1.axhline(0, color='black', linestyle='--', alpha=0.4)
    ax1.fill_between(dates, tw['PAIR_Cum_Ret'], 0,
                     where=(tw['PAIR_Cum_Ret'] >= 0),
                     color='#2ca02c', alpha=0.12, label='Profit Zone')
    ax1.fill_between(dates, tw['PAIR_Cum_Ret'], 0,
                     where=(tw['PAIR_Cum_Ret'] < 0),
                     color='#d62728', alpha=0.12, label='Loss Zone')
    ax1.yaxis.set_major_formatter(pct_fmt)
    ax1.set_title(
        f"{case['title']}\n"
        f"Pairs Trade Cumulative Return  |  Constant β = {const_beta:.4f}  "
        f"(estimated on {metrics['Pre-Entry Days']} pre-entry days)  |  50bps borrow\n"
        f"Final Return: {pair_final:.2f}%  |  Sharpe (Period): {pair_sp:.2f}  "
        f"|  Max DD: {pair_mdd:.2f}%",
        fontweight='bold', pad=12
    )
    ax1.set_ylabel('Cumulative Return (%)')
    ax1.legend(loc='upper left', fontsize=10)

    trans = transforms.blended_transform_factory(ax1.transData, ax1.transAxes)
    ax1.axvline(entry, color='black', linestyle='-.', linewidth=1.5, alpha=0.7)
    ax1.text(entry, 0.95, '  Trade Entry', transform=trans,
             rotation=90, va='top', ha='right',
             fontsize=9, fontweight='bold', color='#333333')

    # ── Panel 2: Individual legs ──────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(dates, tw['LONG_Cum_Ret'],  color='#1f77b4', linewidth=2,
             linestyle='--', label=f"Long {long_ticker} Only  ({long_final:.2f}%)")
    ax2.plot(dates, tw['SHORT_Cum_Ret'], color='#ff7f0e', linewidth=2,
             linestyle='--', label=f"Short {short_ticker} Only ({short_final:.2f}%)")
    ax2.plot(dates, tw['PAIR_Cum_Ret'],  color='#2ca02c', linewidth=2.5,
             label=f"Pairs Trade β={const_beta:.3f} ({pair_final:.2f}%)")
    ax2.axhline(0, color='black', linestyle='--', alpha=0.4)
    ax2.yaxis.set_major_formatter(pct_fmt)
    ax2.set_title('Individual Leg Benchmarks vs. Pairs Strategy',
                  fontweight='bold', pad=8)
    ax2.set_ylabel('Cumulative Return (%)')
    ax2.legend(loc='upper left', fontsize=10)

    # ── Panel 3: Drawdown ─────────────────────────────────────────────
    ax3 = axes[2]
    ax3.fill_between(dates, tw['PAIR_Drawdown_Pct'], 0, color='#d62728', alpha=0.4)
    ax3.plot(dates, tw['PAIR_Drawdown_Pct'], color='#d62728', linewidth=1.5,
             label='Drawdown')
    ax3.axhline(pair_mdd, color='darkred', linestyle=':', linewidth=1.5,
                label=f'Max Drawdown: {pair_mdd:.2f}%')
    ax3.yaxis.set_major_formatter(pct_fmt)
    ax3.set_title('Pairs Strategy Drawdown (%)', fontweight='bold', pad=8)
    ax3.set_ylabel('Drawdown (%)')
    ax3.legend(loc='lower left', fontsize=10)

    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout(h_pad=2.5)
    path = f"{OUTPUT_DIR}/{case_key}_3_backtest_pnl.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [3/3] Saved backtest PnL        → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: DYNAMIC MOMENTUM PAIRS TRADE
# ══════════════════════════════════════════════════════════════════════════════
#
# Motivation: The static strategy (always Long LONG / Short SHORT) fails badly
# in cases where the market rewards the "winner" instead of the "loser" — i.e.,
# when the Winner's Curse thesis inverts. Rather than assuming direction, this
# strategy waits for the spread to reveal which way momentum is actually running
# and trades that direction.
#
# Signal logic (z-score of β-adjusted spread, 20-day rolling window):
#   +2σ breach  → enter LONG the spread (Long LONG / Short SHORT)
#   -2σ breach  → enter SHORT the spread (Short LONG / Long SHORT) = inversion
#   Cross 0     → exit current position
#   New signal in opposite direction while in position → flip
#
# Position sizing: same $100k notional as static strategy.
# Borrow cost: applied on the short leg regardless of direction.
# Direction is determined day-to-day from the previous day's z-score
# (lagged by 1 to avoid look-ahead bias).
# ══════════════════════════════════════════════════════════════════════════════

def run_dynamic_backtest(df, entry_date, zscore_entry=2.0, zscore_exit=0.0):
    """
    Dynamic momentum pairs trade: direction determined by rolling spread z-score.

    Parameters
    ----------
    df            : full DataFrame (fetch_start → fetch_end)
    entry_date    : str  — trade window opens from this date
    zscore_entry  : float — z-score threshold to open/flip a position (default ±2)
    zscore_exit   : float — z-score threshold to exit a position (default 0)

    Returns
    -------
    tw      : trade-window DataFrame with dynamic PnL columns
    metrics : dict of performance metrics (same keys as static strategy)
    """
    entry = pd.to_datetime(entry_date)
    df    = df.copy()

    # ── Constant beta (same OLS method as static strategy) ───────────
    pre      = df[df['Date'] < entry].copy()
    pre_lr   = pre['LONG'].pct_change().dropna()
    pre_sr   = pre['SHORT'].pct_change().dropna()
    aligned  = pd.concat([pre_lr, pre_sr], axis=1).dropna()
    aligned.columns = ['L', 'S']
    const_beta = (aligned['L'].cov(aligned['S']) / aligned['S'].var()
                  if len(aligned) >= 2 else 1.0)

    # ── Z-score computed on FULL history so the rolling window is warm ─
    df['_lr'] = np.log(df['LONG']  / df['LONG'].shift(1))
    df['_sr'] = np.log(df['SHORT'] / df['SHORT'].shift(1))
    spread    = df['_lr'] - const_beta * df['_sr']
    s_mean    = spread.rolling(HEDGE_WINDOW).mean()
    s_std     = spread.rolling(HEDGE_WINDOW).std()
    df['ZScore'] = (spread - s_mean) / s_std

    # ── Trade window ──────────────────────────────────────────────────
    tw = df[df['Date'] >= entry].copy().reset_index(drop=True)
    if tw.empty:
        return None, None

    tw['LONG_Ret']  = tw['LONG'].pct_change().fillna(0)
    tw['SHORT_Ret'] = tw['SHORT'].pct_change().fillna(0)
    daily_borrow    = BORROW_COST / TRADING_DAYS

    # ── Signal generation (lagged by 1 — no look-ahead) ──────────────
    # Position: +1 = Long LONG / Short SHORT, -1 = Short LONG / Long SHORT, 0 = flat
    z         = tw['ZScore'].shift(1).fillna(0)   # yesterday's z-score
    position  = np.zeros(len(tw))
    current   = 0

    for i in range(len(tw)):
        zi = z.iloc[i]
        if current == 0:
            if zi >= zscore_entry:
                current = 1
            elif zi <= -zscore_entry:
                current = -1
        elif current == 1:
            if zi <= zscore_exit:
                current = 0
            elif zi <= -zscore_entry:   # hard flip
                current = -1
        elif current == -1:
            if zi >= -zscore_exit:
                current = 0
            elif zi >= zscore_entry:    # hard flip
                current = 1
        position[i] = current

    tw['Position'] = position

    # ── Daily P&L ─────────────────────────────────────────────────────
    # When Position = +1:  Long LONG, Short SHORT  (standard thesis)
    # When Position = -1:  Short LONG, Long SHORT  (inverted thesis)
    # Borrow cost applies on whichever leg is short, scaled by beta
    tw['DYN_Daily_PnL'] = (
        tw['Position'] * (
            CAPITAL * tw['LONG_Ret']
            - CAPITAL * const_beta * tw['SHORT_Ret']
        )
        - abs(tw['Position']) * CAPITAL * const_beta * daily_borrow
    )

    tw['DYN_Cum_Ret']      = (tw['DYN_Daily_PnL'].cumsum() / CAPITAL) * 100
    tw['DYN_Peak_Ret']     = tw['DYN_Cum_Ret'].cummax()
    tw['DYN_Drawdown_Pct'] = tw['DYN_Cum_Ret'] - tw['DYN_Peak_Ret']

    # ── Metrics ───────────────────────────────────────────────────────
    def _sharpe(pnl, scale):
        s = pnl.iloc[1:]
        e = (s / CAPITAL) - (RISK_FREE / TRADING_DAYS)
        return (e.mean() / e.std()) * np.sqrt(scale) if e.std() != 0 else np.nan

    days_in_market = (tw['Position'] != 0).sum()

    metrics_dyn = {
        'Final Ret %':      tw['DYN_Cum_Ret'].iloc[-1],
        'Max Drawdown %':   (tw['DYN_Cum_Ret'] - tw['DYN_Cum_Ret'].cummax()).min(),
        'Sharpe (Period)':  _sharpe(tw['DYN_Daily_PnL'], len(tw) - 1),
        'Sharpe (Annual)':  _sharpe(tw['DYN_Daily_PnL'], TRADING_DAYS),
        'Win Rate':         (tw['DYN_Daily_PnL'][tw['Position'] != 0] > 0).mean() * 100
                            if days_in_market > 0 else np.nan,
        'Days in Market':   int(days_in_market),
        'Long Signals':     int((tw['Position'] == 1).sum()),
        'Short Signals':    int((tw['Position'] == -1).sum()),
        'Const Beta':       const_beta,
        'ZScore Entry':     zscore_entry,
        'ZScore Exit':      zscore_exit,
    }

    return tw, metrics_dyn


def chart_dynamic_comparison(tw, metrics_static, metrics_dyn, case, case_key):
    """
    Chart 4 of 4: 3-panel comparison of static vs dynamic strategy.

      Panel 1 — Cumulative return: static pairs vs dynamic momentum
      Panel 2 — Position signal over time (+1 / 0 / -1)
      Panel 3 — Drawdown comparison
    """
    if tw is None or metrics_dyn is None:
        print(f"  [4/4] Skipped — dynamic backtest returned no data.")
        return

    dates      = tw['Date']
    pct_fmt    = plt.FuncFormatter(lambda v, _: f'{v:.1f}%')
    entry      = pd.to_datetime(case['entry_date'])

    static_final = metrics_static['Final Ret %']
    dyn_final    = metrics_dyn['Final Ret %']
    dyn_sharpe   = metrics_dyn['Sharpe (Period)']
    dyn_mdd      = metrics_dyn['Max Drawdown %']
    beta         = metrics_dyn['Const Beta']

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 14),
        gridspec_kw={'height_ratios': [3, 1.5, 1.5]},
        sharex=True
    )

    # ── Panel 1: Return comparison ────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(dates, tw['PAIR_Cum_Ret'], color='#1f77b4', linewidth=2,
             linestyle='--', label=f"Static (thesis-fixed): {static_final:.2f}%",
             alpha=0.7)
    ax1.plot(dates, tw['DYN_Cum_Ret'], color='#2ca02c', linewidth=2.5,
             label=f"Dynamic (momentum-following): {dyn_final:.2f}%")
    ax1.axhline(0, color='black', linestyle='--', alpha=0.4)
    ax1.fill_between(dates, tw['DYN_Cum_Ret'], 0,
                     where=(tw['DYN_Cum_Ret'] >= 0),
                     color='#2ca02c', alpha=0.10)
    ax1.fill_between(dates, tw['DYN_Cum_Ret'], 0,
                     where=(tw['DYN_Cum_Ret'] < 0),
                     color='#d62728', alpha=0.10)
    ax1.yaxis.set_major_formatter(pct_fmt)
    ax1.set_title(
        f"{case['title']}\n"
        f"Static vs Dynamic Momentum Strategy  |  β = {beta:.4f}  "
        f"|  Entry threshold: ±{metrics_dyn['ZScore Entry']}σ\n"
        f"Dynamic — Final: {dyn_final:.2f}%  |  Sharpe: {dyn_sharpe:.2f}  "
        f"|  Max DD: {dyn_mdd:.2f}%  "
        f"|  Days in market: {metrics_dyn['Days in Market']}",
        fontweight='bold', pad=12
    )
    ax1.set_ylabel('Cumulative Return (%)')
    ax1.legend(loc='upper left', fontsize=10)

    trans = transforms.blended_transform_factory(ax1.transData, ax1.transAxes)
    ax1.axvline(entry, color='black', linestyle='-.', linewidth=1.5, alpha=0.6)
    ax1.text(entry, 0.95, '  Trade Window Opens', transform=trans,
             rotation=90, va='top', ha='right',
             fontsize=9, fontweight='bold', color='#333333')

    # ── Panel 2: Position signal ──────────────────────────────────────
    ax2 = axes[1]
    long_ticker  = case['long_ticker']
    short_ticker = case['short_ticker']

    ax2.fill_between(dates, tw['Position'], 0,
                     where=(tw['Position'] > 0),
                     color='#2ca02c', alpha=0.5,
                     label=f'Long {long_ticker} / Short {short_ticker}')
    ax2.fill_between(dates, tw['Position'], 0,
                     where=(tw['Position'] < 0),
                     color='#d62728', alpha=0.5,
                     label=f'Short {long_ticker} / Long {short_ticker} (inverted)')
    ax2.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['Inverted', 'Flat', 'Standard'])
    ax2.set_title(
        f"Position Signal  |  "
        f"Standard: {metrics_dyn['Long Signals']} days  |  "
        f"Inverted: {metrics_dyn['Short Signals']} days  |  "
        f"Flat: {len(tw) - metrics_dyn['Days in Market']} days",
        fontweight='bold', pad=8
    )
    ax2.set_ylabel('Position')
    ax2.legend(loc='upper right', fontsize=9)

    # ── Panel 3: Drawdown comparison ─────────────────────────────────
    ax3 = axes[2]
    static_dd = tw['PAIR_Drawdown_Pct']
    ax3.plot(dates, static_dd, color='#1f77b4', linewidth=1.5,
             linestyle='--', alpha=0.7,
             label=f"Static DD (max: {metrics_static['Max Drawdown %']:.2f}%)")
    ax3.fill_between(dates, tw['DYN_Drawdown_Pct'], 0,
                     color='#d62728', alpha=0.35)
    ax3.plot(dates, tw['DYN_Drawdown_Pct'], color='#d62728', linewidth=1.5,
             label=f"Dynamic DD (max: {dyn_mdd:.2f}%)")
    ax3.yaxis.set_major_formatter(pct_fmt)
    ax3.axhline(0, color='black', linewidth=0.8, alpha=0.4)
    ax3.set_title('Drawdown Comparison: Static vs Dynamic (%)',
                  fontweight='bold', pad=8)
    ax3.set_ylabel('Drawdown (%)')
    ax3.legend(loc='lower left', fontsize=9)

    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout(h_pad=2.5)
    path = f"{OUTPUT_DIR}/{case_key}_4_dynamic_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [4/4] Saved dynamic comparison  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-CASE SUMMARY CHART
# ══════════════════════════════════════════════════════════════════════════════

def chart_cross_case_summary(all_metrics, all_metrics_dyn):
    """
    2×2 bar chart comparing Static vs Dynamic strategy across all cases.
    Panels: Final Return, Sharpe, Max Drawdown, Days in Market.
    """
    labels  = list(all_metrics.keys())
    pct_fmt = plt.FuncFormatter(lambda v, _: f'{v:.1f}%')

    stat_rets = [all_metrics[k]['Final Ret %']         for k in labels]
    dyn_rets  = [all_metrics_dyn[k]['Final Ret %']     for k in labels if k in all_metrics_dyn]
    stat_sh   = [all_metrics[k]['Sharpe (Period)']     for k in labels]
    dyn_sh    = [all_metrics_dyn[k]['Sharpe (Period)'] for k in labels if k in all_metrics_dyn]
    stat_mdd  = [abs(all_metrics[k]['Max Drawdown %'])      for k in labels]
    dyn_mdd   = [abs(all_metrics_dyn[k]['Max Drawdown %'])  for k in labels if k in all_metrics_dyn]
    dyn_dim   = [all_metrics_dyn[k]['Days in Market']  for k in labels if k in all_metrics_dyn]

    display_labels = {
        'anadarko': 'Anadarko\n(2019)',
        'hulu':     'Hulu\n(2023)',
        'ksu':      'KSU\n(2021)',
    }
    x_labels = [display_labels.get(k, k) for k in labels]
    x        = np.arange(len(labels))
    w        = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()

    # Panel A — Final Return
    ax1 = axes[0]
    b1 = ax1.bar(x - w/2, stat_rets, w, label='Static',  color='#1f77b4', alpha=0.85)
    b2 = ax1.bar(x + w/2, dyn_rets,  w, label='Dynamic', color='#2ca02c', alpha=0.85)
    ax1.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax1.yaxis.set_major_formatter(pct_fmt)
    for bar, val in list(zip(b1, stat_rets)) + list(zip(b2, dyn_rets)):
        ypos = bar.get_height() + (0.3 if bar.get_height() >= 0 else -1.2)
        ax1.text(bar.get_x() + bar.get_width()/2, ypos,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_xticks(x); ax1.set_xticklabels(x_labels, fontsize=10)
    ax1.set_title('Final Return (%)', fontweight='bold', pad=10)
    ax1.set_ylabel('Cumulative Return (%)')
    ax1.legend(fontsize=10)

    # Panel B — Sharpe
    ax2 = axes[1]
    b3 = ax2.bar(x - w/2, stat_sh, w, label='Static',  color='#1f77b4', alpha=0.85)
    b4 = ax2.bar(x + w/2, dyn_sh,  w, label='Dynamic', color='#2ca02c', alpha=0.85)
    ax2.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax2.axhline(1, color='gray',  linewidth=1, linestyle='--', alpha=0.6, label='Sharpe=1')
    for bar, val in list(zip(b3, stat_sh)) + list(zip(b4, dyn_sh)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_xticks(x); ax2.set_xticklabels(x_labels, fontsize=10)
    ax2.set_title('Sharpe Ratio (Period)', fontweight='bold', pad=10)
    ax2.set_ylabel('Sharpe Ratio')
    ax2.legend(fontsize=10)

    # Panel C — Max Drawdown
    ax3 = axes[2]
    b5 = ax3.bar(x - w/2, stat_mdd, w, label='Static',  color='#1f77b4', alpha=0.85)
    b6 = ax3.bar(x + w/2, dyn_mdd,  w, label='Dynamic', color='#2ca02c', alpha=0.85)
    ax3.yaxis.set_major_formatter(pct_fmt)
    all_mdd = stat_mdd + dyn_mdd
    for bar, val in list(zip(b5, stat_mdd)) + list(zip(b6, dyn_mdd)):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (max(all_mdd) * 0.02),
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax3.set_xticks(x); ax3.set_xticklabels(x_labels, fontsize=10)
    ax3.set_title('Max Drawdown (%, positive)', fontweight='bold', pad=10)
    ax3.set_ylabel('Max Drawdown (%)')
    ax3.legend(fontsize=10)

    # Panel D — Days in market (dynamic only)
    ax4 = axes[3]
    b7 = ax4.bar(x, dyn_dim, color='#9467bd', alpha=0.85, edgecolor='white')
    for bar, val in zip(b7, dyn_dim):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax4.set_xticks(x); ax4.set_xticklabels(x_labels, fontsize=10)
    ax4.set_title('Dynamic: Days in Market\n(out of total trade window)',
                  fontweight='bold', pad=10)
    ax4.set_ylabel('Trading Days')

    fig.suptitle(
        "Winner's Curse Pairs Trade — Static vs Dynamic Momentum Strategy\n"
        "Static: fixed Long/Short at catalyst  |  Dynamic: z-score momentum signal (±2σ)",
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/summary_static_vs_dynamic.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved static vs dynamic summary → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    setup_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_metrics     = {}
    all_metrics_dyn = {}

    for case_key, case in CASES.items():
        print(f"\n{'═'*60}")
        print(f"  Processing: {case['title']}")
        print(f"{'═'*60}")

        # 1. Fetch data
        df = fetch_data(
            case['long_ticker'],
            case['short_ticker'],
            case['fetch_start'],
            case['fetch_end'],
        )

        # 2. Run static backtest
        tw, metrics = run_backtest(df, case['entry_date'])

        if metrics:
            all_metrics[case_key] = metrics
            print(f"\n  [Static] Pairs Trade results:")
            print(f"    Final Return:    {metrics['Final Ret %']:>10.2f}%")
            print(f"    Sharpe (Period):  {metrics['Sharpe (Period)']:>10.3f}")
            print(f"    Sharpe (Annual):  {metrics['Sharpe (Annual)']:>10.3f}")
            print(f"    Max Drawdown:    {metrics['Max Drawdown %']:>10.2f}%")
            print(f"    Win Rate:         {metrics['Win Rate']:>9.1f}%")
            print(f"    Constant β:       {metrics['Const Beta']:>10.4f}  "
                  f"(OLS on {metrics['Pre-Entry Days']} pre-entry days)")

        # 3. Run dynamic backtest
        tw_dyn, metrics_dyn = run_dynamic_backtest(df, case['entry_date'])

        if metrics_dyn:
            all_metrics_dyn[case_key] = metrics_dyn
            print(f"\n  [Dynamic] Momentum strategy results:")
            print(f"    Final Return:    {metrics_dyn['Final Ret %']:>10.2f}%")
            print(f"    Sharpe (Period):  {metrics_dyn['Sharpe (Period)']:>10.3f}")
            print(f"    Sharpe (Annual):  {metrics_dyn['Sharpe (Annual)']:>10.3f}")
            print(f"    Max Drawdown:    {metrics_dyn['Max Drawdown %']:>10.2f}%")
            print(f"    Win Rate:         {metrics_dyn['Win Rate']:>9.1f}%")
            print(f"    Days in Market:   {metrics_dyn['Days in Market']:>9}  "
                  f"({metrics_dyn['Long Signals']} standard / "
                  f"{metrics_dyn['Short Signals']} inverted)")

        # 4. Generate all four charts
        print(f"\n  Generating charts...")
        chart_normalized_returns(df, case, case_key)
        chart_spread_zscore(df, case, case_key)
        chart_backtest_pnl(df, tw, metrics, case, case_key)
        # Pass tw with dynamic columns merged in (tw already has PAIR_Cum_Ret
        # from static backtest; we merge dynamic columns onto it)
        if tw is not None and tw_dyn is not None:
            tw['DYN_Daily_PnL']   = tw_dyn['DYN_Daily_PnL'].values
            tw['DYN_Cum_Ret']     = tw_dyn['DYN_Cum_Ret'].values
            tw['DYN_Drawdown_Pct']= tw_dyn['DYN_Drawdown_Pct'].values
            tw['Position']        = tw_dyn['Position'].values
            chart_dynamic_comparison(tw, metrics, metrics_dyn, case, case_key)

    # 5. Cross-case summary
    if len(all_metrics) == len(CASES) and len(all_metrics_dyn) == len(CASES):
        chart_cross_case_summary(all_metrics, all_metrics_dyn)

    # 6. Final summary table
    print(f"\n{'═'*70}")
    print(f"  FINAL CROSS-CASE SUMMARY")
    print(f"{'═'*70}")
    print(f"  {'Case':<36} {'Strategy':<10} {'Return':>8} {'Sharpe':>8} {'Max DD':>8} {'Win%':>7}")
    print(f"  {'-'*77}")
    labels_display = {
        'anadarko': 'Anadarko 2019 — CVX / OXY',
        'hulu':     'Hulu     2023 — CMCSA / DIS',
        'ksu':      'KSU      2021 — CNI / CP  ',
    }
    for k, lbl in labels_display.items():
        if k in all_metrics:
            m  = all_metrics[k]
            md = all_metrics_dyn.get(k, {})
            print(f"  {lbl:<36} {'Static':<10} "
                  f"{m['Final Ret %']:>7.2f}%  "
                  f"{m['Sharpe (Period)']:>7.2f}  "
                  f"{m['Max Drawdown %']:>7.2f}%  "
                  f"{m['Win Rate']:>5.1f}%")
            if md:
                print(f"  {'':36} {'Dynamic':<10} "
                      f"{md['Final Ret %']:>7.2f}%  "
                      f"{md['Sharpe (Period)']:>7.2f}  "
                      f"{md['Max Drawdown %']:>7.2f}%  "
                      f"{md['Win Rate']:>5.1f}%")
            print(f"  {'-'*77}")

    print(f"\n  All charts saved to '{OUTPUT_DIR}/'")
    print(f"  Files generated per case:")
    print(f"    <case>_1_normalized_returns.png")
    print(f"    <case>_2_spread_zscore.png")
    print(f"    <case>_3_backtest_pnl.png        (static strategy)")
    print(f"    <case>_4_dynamic_comparison.png  (static vs dynamic)")
    print(f"    summary_static_vs_dynamic.png    (cross-case 2×2 summary)")


if __name__ == '__main__':
    main()