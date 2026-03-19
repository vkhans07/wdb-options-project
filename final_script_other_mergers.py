"""
generate_validation_charts.py
═══════════════════════════════════════════════════════════════════════════════
Unified Winner's Curse pairs trade validation script.

Runs FOUR strategies across THREE real-world M&A bidding war cases and
produces all charts. Strategies are the 2×2 combination of:

  Hedge ratio  ×  Entry logic
  ─────────────────────────────────────────────────────────────────────
  CONSTANT β   │ OLS on pre-entry window, held fixed for entire trade
  ROLLING β    │ 20-day Cov/Var, updated daily (lagged 1 day)
  ─────────────────────────────────────────────────────────────────────
  STATIC entry │ Enter at catalyst, hold to end, always Long/Short
  DYNAMIC entry│ Enter when z-score breaks ±2σ, exit at 0, can invert
  ─────────────────────────────────────────────────────────────────────

Strategy IDs:
  S1 — Const β / Static    (thesis-fixed baseline)
  S2 — Const β / Dynamic   (momentum signal, stable sizing)
  S3 — Rolling β / Static  (adaptive sizing, fixed direction)
  S4 — Rolling β / Dynamic (fully adaptive)

Cases:
  ① Anadarko (2019)  CVX vs OXY
  ② Hulu     (2023)  CMCSA vs DIS
  ③ KSU      (2021)  CNI vs CP

Charts per case (→ charts/validation/):
  <case>_1_normalized_returns.png
  <case>_2_spread_zscore.png          (const-β panel + rolling-β panel)
  <case>_3_all_strategies.png         (all 4 strategies overlaid)
  <case>_4_dynamic_signals.png        (position signals for S2 and S4)

Summary charts:
  summary_all_strategies.png          (2×2 bar chart, all cases × all strategies)
  summary_beta_comparison.png         (const vs rolling β, by entry style)

Requirements:
  pip install yfinance statsmodels pandas numpy matplotlib scipy seaborn
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.transforms as transforms
import seaborn as sns
import yfinance as yf


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CASES = {
    'anadarko': {
        'title':        'Anadarko Bidding War (2019): Chevron vs Occidental',
        'long_ticker':   'CVX',
        'short_ticker':  'OXY',
        'long_label':    'CVX (walked away, $1B fee)',
        'short_label':   'OXY (won, $38B debt burden)',
        'fetch_start':   '2019-02-01',
        'fetch_end':     '2019-12-31',
        'entry_date':    '2019-05-05',
        'milestones': {
            '2019-04-12': 'Chevron $33B bid',
            '2019-04-24': 'OXY $38B counterbid',
            '2019-05-05': 'Chevron walks ← ENTRY',
            '2019-08-08': 'Deal closes',
        },
    },
    'hulu': {
        'title':        'Hulu Buyout (2023): Disney vs Comcast',
        'long_ticker':   'CMCSA',
        'short_ticker':  'DIS',
        'long_label':    'CMCSA (walked away with cash)',
        'short_label':   'DIS (absorbed ~$8.6B Hulu liability)',
        'fetch_start':   '2023-08-01',
        'fetch_end':     '2024-06-30',
        'entry_date':    '2023-11-01',
        'milestones': {
            '2023-09-06': 'Disney invokes put option',
            '2023-11-01': 'Valuation agreed ← ENTRY',
            '2024-02-01': 'Disney full ownership complete',
        },
    },
    'ksu': {
        'title':        'KSU Railway War (2021): CP vs CN',
        'long_ticker':   'CNI',
        'short_ticker':  'CP',
        'long_label':    'CNI (walked away, $700M fee)',
        'short_label':   'CP (won KSU, heavy debt)',
        'fetch_start':   '2021-05-01',
        'fetch_end':     '2022-09-15',
        'entry_date':    '2021-09-15',
        'milestones': {
            '2021-03-21': 'CP initial $25B bid',
            '2021-05-13': 'CNI $33.7B counterbid',
            '2021-09-15': 'STB blocks CNI ← ENTRY',
            '2022-04-14': 'CP-KSU merger closes',
        },
    },
}

CAPITAL      = 100_000
RISK_FREE    = 0.045
TRADING_DAYS = 252
BORROW_COST  = 0.0050
HEDGE_WINDOW = 20
ZSCORE_ENTRY = 2.0   # breach ±this to open/flip position
ZSCORE_EXIT  = 0.5   # must revert inside ±this to exit — 0.5 keeps you in longer
OUTPUT_DIR   = 'charts/validation'

COLORS = {
    'S1': '#1f77b4',
    'S2': '#ff7f0e',
    'S3': '#2ca02c',
    'S4': '#9467bd',
}
LABELS = {
    'S1': 'Const β / Static',
    'S2': 'Const β / Dynamic',
    'S3': 'Rolling β / Static',
    'S4': 'Rolling β / Dynamic',
}


# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
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


def pct_fmt():
    return plt.FuncFormatter(lambda v, _: f'{v:.1f}%')


def format_x_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def add_milestones(ax, milestones):
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for date_str, label in milestones.items():
        d = pd.to_datetime(date_str)
        ax.axvline(d, color='black', linestyle='-.', alpha=0.6, linewidth=1.5)
        ax.text(d, 0.95, f' {label}', transform=trans,
                rotation=90, va='top', ha='right',
                fontsize=10, fontweight='bold', color='#333333')


def add_entry_vline(ax, entry_date, label='Trade Entry'):
    d     = pd.to_datetime(entry_date)
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.axvline(d, color='black', linestyle='-.', linewidth=1.5, alpha=0.7)
    ax.text(d, 0.95, f'  {label}', transform=trans,
            rotation=90, va='top', ha='right',
            fontsize=9, fontweight='bold', color='#333333')


def _sharpe(pnl_series, scale):
    s = pnl_series.iloc[1:]
    e = (s / CAPITAL) - (RISK_FREE / TRADING_DAYS)
    return (e.mean() / e.std()) * np.sqrt(scale) if e.std() != 0 else np.nan


def fetch_data(long_ticker, short_ticker, start, end):
    print(f"  Downloading {long_ticker} & {short_ticker} ({start} → {end})...")
    raw   = yf.download([long_ticker, short_ticker],
                        start=start, end=end,
                        auto_adjust=True, progress=False)
    close = raw['Close'][[long_ticker, short_ticker]].copy()
    close.columns = ['LONG', 'SHORT']
    close = (close.reset_index()
                  .rename(columns={'index': 'Date', 'Datetime': 'Date'}))
    close['Date'] = pd.to_datetime(close['Date']).dt.tz_localize(None)
    close = close.sort_values('Date').reset_index(drop=True).ffill().dropna()
    print(f"  {len(close)} trading days loaded.")
    return close


def _rolling_beta_series(df, window=HEDGE_WINDOW):
    lr = df['LONG'].pct_change()
    sr = df['SHORT'].pct_change()
    return (lr.rolling(window).cov(sr) / sr.rolling(window).var()
            ).shift(1).fillna(1.0)


def _const_beta_value(df, entry_date):
    entry = pd.to_datetime(entry_date)
    pre   = df[df['Date'] < entry]
    lr    = pre['LONG'].pct_change().dropna()
    sr    = pre['SHORT'].pct_change().dropna()
    aln   = pd.concat([lr, sr], axis=1).dropna()
    aln.columns = ['L', 'S']
    if len(aln) >= 2:
        return aln['L'].cov(aln['S']) / aln['S'].var(), len(aln)
    return 1.0, len(aln)


def _zscore_series(df, beta, window=HEDGE_WINDOW):
    lr     = np.log(df['LONG']  / df['LONG'].shift(1))
    sr     = np.log(df['SHORT'] / df['SHORT'].shift(1))
    spread = lr - beta * sr
    return (spread - spread.rolling(window).mean()) / spread.rolling(window).std()


def _dynamic_positions(z_lagged):
    pos = np.zeros(len(z_lagged))
    cur = 0
    for i, z in enumerate(z_lagged):
        if cur == 0:
            if   z >=  ZSCORE_ENTRY: cur =  1
            elif z <= -ZSCORE_ENTRY: cur = -1
        elif cur == 1:
            if   z <=  ZSCORE_EXIT:  cur =  0
            elif z <= -ZSCORE_ENTRY: cur = -1
        elif cur == -1:
            if   z >= -ZSCORE_EXIT:  cur =  0
            elif z >=  ZSCORE_ENTRY: cur =  1
        pos[i] = cur
    return pos


def _pair_pnl(tw, beta, positions, long_ret_col='LONG_LogRet',
              short_ret_col='SHORT_LogRet'):
    """
    Daily pair P&L using true daily returns (exp(log_ret) - 1).
    beta can be a scalar (constant) or a column name string (rolling).
    positions is a Series of +1 / -1 / 0.
    """
    bd = BORROW_COST / TRADING_DAYS
    b  = tw[beta] if isinstance(beta, str) else beta
    return (
        positions * (CAPITAL * tw[long_ret_col] - CAPITAL * b * tw[short_ret_col])
        - np.abs(positions) * CAPITAL * b * bd
    )


def _metrics(pnl, cum_ret, pos=None):
    dd = cum_ret - cum_ret.cummax()
    if pos is not None and (pos != 0).sum() > 0:
        wr = (pnl[pos != 0] > 0).mean() * 100
    else:
        wr = (pnl > 0).mean() * 100
    return {
        'Final Ret %':     cum_ret.iloc[-1],
        'Max Drawdown %':  dd.min(),
        'Sharpe (Period)': _sharpe(pnl, len(pnl) - 1),
        'Sharpe (Annual)': _sharpe(pnl, TRADING_DAYS),
        'Win Rate':        wr,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CORE BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_all_backtests(df, entry_date):
    """
    Compute all four strategies on df. Returns (trade_window_df, metrics_dict).

    Trade-window columns:
      LONG_Ret, SHORT_Ret           daily pct returns
      LONG_Cum_Ret, SHORT_Cum_Ret   individual leg cum returns (%)
      Roll_Beta                     rolling beta series
      ZScore_Const, ZScore_Rolling  z-scores under each beta
      S{1..4}_Daily_PnL             daily dollar PnL per strategy
      S{1..4}_Cum_Ret               cumulative return (%) per strategy
      S{1..4}_Drawdown              drawdown (%) per strategy
      S2_Position, S4_Position      position signal for dynamic strategies
    """
    entry = pd.to_datetime(entry_date)
    df    = df.copy()

    cb, n_pre       = _const_beta_value(df, entry_date)
    df['Roll_Beta'] = _rolling_beta_series(df, HEDGE_WINDOW)
    df['ZScore_Const']   = _zscore_series(df, cb,              HEDGE_WINDOW)
    df['ZScore_Rolling'] = _zscore_series(df, df['Roll_Beta'], HEDGE_WINDOW)

    tw = df[df['Date'] >= entry].copy().reset_index(drop=True)
    if tw.empty:
        return None, {}

    # Simple pct returns (kept for beta estimation consistency)
    tw['LONG_Ret']  = tw['LONG'].pct_change().fillna(0)
    tw['SHORT_Ret'] = tw['SHORT'].pct_change().fillna(0)
    # Log returns → exponentiated back to true daily returns for PnL
    # exp(log_return) - 1 == true simple return; avoids Jensen's inequality
    # bias that accumulates when you sum simple returns over a path
    tw['LONG_LogRet']  = (np.log(tw['LONG']  / tw['LONG'].shift(1))
                          .fillna(0).apply(lambda x: np.exp(x) - 1))
    tw['SHORT_LogRet'] = (np.log(tw['SHORT'] / tw['SHORT'].shift(1))
                          .fillna(0).apply(lambda x: np.exp(x) - 1))
    bd = BORROW_COST / TRADING_DAYS

    # Individual legs (reference lines on charts) — using true log returns
    tw['LONG_Daily_PnL']  = CAPITAL * tw['LONG_LogRet']
    tw['SHORT_Daily_PnL'] = -CAPITAL * tw['SHORT_LogRet'] - CAPITAL * bd
    tw['LONG_Cum_Ret']    = (tw['LONG_Daily_PnL'].cumsum()  / CAPITAL) * 100
    tw['SHORT_Cum_Ret']   = (tw['SHORT_Daily_PnL'].cumsum() / CAPITAL) * 100

    # S1 — Const β / Static
    pos_s1 = pd.Series(np.ones(len(tw)))
    tw['S1_Daily_PnL'] = _pair_pnl(tw, cb, pos_s1)

    # S2 — Const β / Dynamic
    pos_s2 = pd.Series(_dynamic_positions(tw['ZScore_Const'].shift(1).fillna(0)))
    tw['S2_Position']  = pos_s2
    tw['S2_Daily_PnL'] = _pair_pnl(tw, cb, pos_s2)

    # S3 — Rolling β / Static
    pos_s3 = pd.Series(np.ones(len(tw)))
    tw['S3_Daily_PnL'] = _pair_pnl(tw, 'Roll_Beta', pos_s3)

    # S4 — Rolling β / Dynamic
    pos_s4 = pd.Series(_dynamic_positions(tw['ZScore_Rolling'].shift(1).fillna(0)))
    tw['S4_Position']  = pos_s4
    tw['S4_Daily_PnL'] = _pair_pnl(tw, 'Roll_Beta', pos_s4)

    # Cumulative returns and drawdowns
    for sid in ('S1', 'S2', 'S3', 'S4'):
        tw[f'{sid}_Cum_Ret']  = (tw[f'{sid}_Daily_PnL'].cumsum() / CAPITAL) * 100
        tw[f'{sid}_Drawdown'] = tw[f'{sid}_Cum_Ret'] - tw[f'{sid}_Cum_Ret'].cummax()

    # Metrics
    m = {}
    m['S1'] = {
        **_metrics(tw['S1_Daily_PnL'], tw['S1_Cum_Ret']),
        'Beta Type': 'Constant', 'Entry Type': 'Static',
        'Const Beta': cb, 'Pre-Entry Days': n_pre,
    }
    m['S2'] = {
        **_metrics(tw['S2_Daily_PnL'], tw['S2_Cum_Ret'], pos=tw['S2_Position']),
        'Beta Type': 'Constant', 'Entry Type': 'Dynamic',
        'Const Beta': cb, 'Pre-Entry Days': n_pre,
        'Days in Market': int((tw['S2_Position'] != 0).sum()),
        'Long Days':      int((tw['S2_Position'] ==  1).sum()),
        'Short Days':     int((tw['S2_Position'] == -1).sum()),
    }
    m['S3'] = {
        **_metrics(tw['S3_Daily_PnL'], tw['S3_Cum_Ret']),
        'Beta Type': 'Rolling', 'Entry Type': 'Static',
        'Beta Entry': tw['Roll_Beta'].iloc[0],
        'Beta Mean':  tw['Roll_Beta'].mean(),
        'Beta Range': f"{tw['Roll_Beta'].min():.3f}–{tw['Roll_Beta'].max():.3f}",
    }
    m['S4'] = {
        **_metrics(tw['S4_Daily_PnL'], tw['S4_Cum_Ret'], pos=tw['S4_Position']),
        'Beta Type': 'Rolling', 'Entry Type': 'Dynamic',
        'Beta Entry': tw['Roll_Beta'].iloc[0],
        'Beta Mean':  tw['Roll_Beta'].mean(),
        'Beta Range': f"{tw['Roll_Beta'].min():.3f}–{tw['Roll_Beta'].max():.3f}",
        'Days in Market': int((tw['S4_Position'] != 0).sum()),
        'Long Days':      int((tw['S4_Position'] ==  1).sum()),
        'Short Days':     int((tw['S4_Position'] == -1).sum()),
    }

    return tw, m


# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — NORMALIZED RETURNS
# ══════════════════════════════════════════════════════════════════════════════

def chart_normalized_returns(df, case, case_key):
    fig, ax = plt.subplots(figsize=(16, 8))
    entry   = pd.to_datetime(case['entry_date'])

    for col, label, color in [
        ('LONG',  case['long_label'],  '#1f77b4'),
        ('SHORT', case['short_label'], '#d62728'),
    ]:
        norm = (df[col] / df[col].iloc[0]) * 100
        ax.plot(df['Date'], norm, label=label, color=color, linewidth=2.5)

    ax.axhline(100, color='black', linestyle=':', alpha=0.4, linewidth=1)
    ax.axvspan(entry, df['Date'].max(), color='#2ca02c', alpha=0.05,
               label='Trade Window Active')
    add_milestones(ax, case['milestones'])

    ax.set_title(
        f"{case['title']}\nPercentage Returns (Base 100 = first trading day of study window)",
        pad=20)
    ax.set_ylabel('Price Return (%)')
    ax.legend(loc='lower left', fontsize=11)
    format_x_axis(ax)
    plt.tight_layout()

    path = f"{OUTPUT_DIR}/{case_key}_1_normalized_returns.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [1/4] Saved normalized returns  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — SPREAD Z-SCORE (both beta types)
# ══════════════════════════════════════════════════════════════════════════════

def chart_spread_zscore(df, case, case_key):
    """
    Two-panel chart: const-β z-score (top) and rolling-β z-score (bottom).
    Shows how the entry signal looks under each beta assumption.
    """
    entry = pd.to_datetime(case['entry_date'])
    cb, _ = _const_beta_value(df, case['entry_date'])
    df    = df.copy()
    df['Roll_Beta'] = _rolling_beta_series(df, HEDGE_WINDOW)
    zc = _zscore_series(df, cb,              HEDGE_WINDOW)
    zr = _zscore_series(df, df['Roll_Beta'], HEDGE_WINDOW)

    fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

    def _panel(ax, zscore, title, color):
        ax.plot(df['Date'], zscore, color=color, linewidth=2,
                label='Spread Z-Score', zorder=3)
        for lvl, sty, lbl in [(1, ':', '±1σ'), (2, '--', '±2σ entry band')]:
            ax.axhline( lvl, color='gray', linestyle=sty, alpha=0.7,
                        linewidth=1.2, label=lbl if lvl == 2 else None)
            ax.axhline(-lvl, color='gray', linestyle=sty, alpha=0.7,
                        linewidth=1.2)
        ax.axhline(0, color='black', linestyle='-', alpha=0.2, linewidth=1)
        ax.axvspan(entry, df['Date'].max(), color='#2ca02c', alpha=0.06,
                   label='Trade Window Active')
        add_entry_vline(ax, entry)
        post = zscore[df['Date'] >= entry].dropna()
        if not post.empty:
            idx = post.abs().idxmax()
            pz  = zscore.loc[idx]
            pd_ = df.loc[idx, 'Date']
            ax.annotate(
                f'Peak divergence\nZ = {pz:.2f}σ',
                xy=(pd_, pz),
                xytext=(pd_, pz + (1.1 if pz > 0 else -1.1)),
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
                fontsize=10, color='#d62728', fontweight='bold', ha='center',
            )
        ax.set_ylabel('Z-Score (σ)')
        ax.legend(loc='upper left', fontsize=10)
        ax.set_title(title, pad=10)

    _panel(axes[0], zc,
           f"{case['title']} — Spread Z-Score | Constant β = {cb:.4f}\n"
           f"β-Adjusted log-return spread, 20-day rolling normalisation",
           '#1f77b4')
    _panel(axes[1], zr,
           f"{case['title']} — Spread Z-Score | Rolling β (20-day)\n"
           f"β-Adjusted log-return spread, 20-day rolling normalisation",
           '#2ca02c')

    format_x_axis(axes[1])
    plt.tight_layout(h_pad=2)
    path = f"{OUTPUT_DIR}/{case_key}_2_spread_zscore.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [2/4] Saved spread z-score      → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 3a — PER-STRATEGY INDIVIDUAL CHARTS (one file per strategy per case)
# ══════════════════════════════════════════════════════════════════════════════

def chart_per_strategy(tw, metrics, case, case_key):
    """
    Generates one dedicated chart per strategy (4 files per case):
      <case>_S1_const_static.png
      <case>_S2_const_dynamic.png
      <case>_S3_roll_static.png
      <case>_S4_roll_dynamic.png

    Each chart has:
      Panel 1 — Cumulative return (%) with profit/loss shading and individual legs
      Panel 2 — Drawdown (%)
      Panel 3 — Position signal (dynamic strategies only; skipped for S1/S3)
    """
    if tw is None:
        return

    dates  = tw['Date']
    entry  = pd.to_datetime(case['entry_date'])
    fmt    = pct_fmt()
    lt, st = case['long_ticker'], case['short_ticker']

    sid_filenames = {
        'S1': 'S1_const_static',
        'S2': 'S2_const_dynamic',
        'S3': 'S3_roll_static',
        'S4': 'S4_roll_dynamic',
    }

    for sid, fname in sid_filenames.items():
        m         = metrics[sid]
        is_dyn    = m['Entry Type'] == 'Dynamic'
        n_panels  = 3 if is_dyn else 2
        ratios    = [3, 1.5, 1] if is_dyn else [3, 1.5]

        fig, axes = plt.subplots(n_panels, 1, figsize=(16, 5 * n_panels),
                                 gridspec_kw={'height_ratios': ratios},
                                 sharex=True)
        if n_panels == 2:
            axes = list(axes)  # ensure indexable

        # ── Panel 1: Cumulative return ────────────────────────────────
        ax1 = axes[0]
        ax1.plot(dates, tw['LONG_Cum_Ret'],  color='#aec7e8', linewidth=1.2,
                 linestyle=':', label=f'Long {lt} only  '
                                      f'({tw["LONG_Cum_Ret"].iloc[-1]:.1f}%)')
        ax1.plot(dates, tw['SHORT_Cum_Ret'], color='#ffbb78', linewidth=1.2,
                 linestyle=':', label=f'Short {st} only  '
                                      f'({tw["SHORT_Cum_Ret"].iloc[-1]:.1f}%)')
        ax1.plot(dates, tw[f'{sid}_Cum_Ret'], color=COLORS[sid], linewidth=2.5,
                 label=f"{LABELS[sid]}  ({m['Final Ret %']:.1f}%)")
        ax1.axhline(0, color='black', linestyle='--', alpha=0.4)
        ax1.fill_between(dates, tw[f'{sid}_Cum_Ret'], 0,
                         where=(tw[f'{sid}_Cum_Ret'] >= 0),
                         color=COLORS[sid], alpha=0.10)
        ax1.fill_between(dates, tw[f'{sid}_Cum_Ret'], 0,
                         where=(tw[f'{sid}_Cum_Ret'] < 0),
                         color='#d62728', alpha=0.10)
        ax1.yaxis.set_major_formatter(fmt)
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.legend(loc='upper left', fontsize=10)
        add_entry_vline(ax1, entry)

        # Build beta info string for title
        if m['Beta Type'] == 'Constant':
            beta_str = f"Constant β = {m['Const Beta']:.4f}  ({m['Pre-Entry Days']} pre-entry days)"
        else:
            beta_str = (f"Rolling β (20-day)  entry={m['Beta Entry']:.3f}  "
                        f"mean={m['Beta Mean']:.3f}  range={m['Beta Range']}")

        if is_dyn:
            entry_str = (f"Dynamic entry (±{ZSCORE_ENTRY}σ / exit {ZSCORE_EXIT}σ)  "
                         f"in-market={m['Days in Market']}d  "
                         f"({m['Long Days']} std / {m['Short Days']} inv)")
        else:
            entry_str = "Static entry — hold from catalyst date"

        ax1.set_title(
            f"{case['title']} — {LABELS[sid]}\n"
            f"{beta_str}\n"
            f"{entry_str}\n"
            f"Return: {m['Final Ret %']:.2f}%  |  "
            f"Sharpe (Period): {m['Sharpe (Period)']:.2f}  |  "
            f"Sharpe (Annual): {m['Sharpe (Annual)']:.2f}  |  "
            f"Max DD: {m['Max Drawdown %']:.2f}%  |  "
            f"Win Rate: {m['Win Rate']:.1f}%",
            pad=12
        )

        # ── Panel 2: Drawdown ─────────────────────────────────────────
        ax2 = axes[1]
        ax2.fill_between(dates, tw[f'{sid}_Drawdown'], 0,
                         color='#d62728', alpha=0.35)
        ax2.plot(dates, tw[f'{sid}_Drawdown'], color='#d62728', linewidth=1.5,
                 label=f"Drawdown (max: {m['Max Drawdown %']:.2f}%)")
        ax2.axhline(m['Max Drawdown %'], color='darkred', linestyle=':',
                    linewidth=1.2, label=f"Max DD: {m['Max Drawdown %']:.2f}%")
        ax2.axhline(0, color='black', linewidth=0.8, alpha=0.4)
        ax2.yaxis.set_major_formatter(fmt)
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend(loc='lower left', fontsize=10)
        ax2.set_title('Drawdown (%)', fontweight='bold', pad=6)

        # ── Panel 3: Position signal (dynamic only) ───────────────────
        if is_dyn:
            ax3 = axes[2]
            pos = tw[f'{sid}_Position']
            ax3.fill_between(dates, pos, 0, where=(pos > 0),
                             color='#2ca02c', alpha=0.55,
                             label=f'Standard: Long {lt} / Short {st}')
            ax3.fill_between(dates, pos, 0, where=(pos < 0),
                             color='#d62728', alpha=0.55,
                             label=f'Inverted: Short {lt} / Long {st}')
            ax3.axhline(0, color='black', linewidth=0.8, alpha=0.5)
            ax3.set_yticks([-1, 0, 1])
            ax3.set_yticklabels(['Inverted', 'Flat', 'Standard'], fontsize=9)
            ax3.set_ylabel('Position')
            ax3.legend(loc='upper right', fontsize=10)
            ax3.set_title('Position Signal (lagged 1-day z-score)',
                          fontweight='bold', pad=6)

        # X-axis on bottom panel
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout(h_pad=2)
        path = f"{OUTPUT_DIR}/{case_key}_{fname}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"       Saved {LABELS[sid]:<24} → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 3b — ALL FOUR STRATEGIES OVERLAID
# ══════════════════════════════════════════════════════════════════════════════

def chart_all_strategies(tw, metrics, case, case_key):
    """
    Two-panel chart.
    Panel 1: Cumulative return (%) — all 4 strategies + individual legs.
    Panel 2: Drawdown (%) — all 4 strategies.
    """
    if tw is None:
        print(f"  [3/4] Skipped — no trade window data.")
        return

    dates = tw['Date']
    entry = pd.to_datetime(case['entry_date'])
    fmt   = pct_fmt()

    fig, axes = plt.subplots(2, 1, figsize=(16, 12),
                             gridspec_kw={'height_ratios': [2.5, 1.5]},
                             sharex=True)

    # Panel 1 — cumulative returns
    ax1 = axes[0]
    ax1.plot(dates, tw['LONG_Cum_Ret'],  color='#aec7e8', linewidth=1.2,
             linestyle=':', label=f"Long {case['long_ticker']} only "
                                  f"({tw['LONG_Cum_Ret'].iloc[-1]:.1f}%)")
    ax1.plot(dates, tw['SHORT_Cum_Ret'], color='#ffbb78', linewidth=1.2,
             linestyle=':', label=f"Short {case['short_ticker']} only "
                                  f"({tw['SHORT_Cum_Ret'].iloc[-1]:.1f}%)")
    for sid in ('S1', 'S2', 'S3', 'S4'):
        m   = metrics[sid]
        lbl = (f"{LABELS[sid]}  "
               f"({m['Final Ret %']:.1f}%  Sharpe={m['Sharpe (Period)']:.2f})")
        ax1.plot(dates, tw[f'{sid}_Cum_Ret'],
                 color=COLORS[sid], linewidth=2.5, label=lbl)
    ax1.axhline(0, color='black', linestyle='--', alpha=0.4)
    ax1.yaxis.set_major_formatter(fmt)
    ax1.set_ylabel('Cumulative Return (%)')
    ax1.legend(loc='upper left', fontsize=9)
    add_entry_vline(ax1, entry)
    ax1.set_title(
        f"{case['title']}\nAll Four Strategies — Cumulative Return (%)\n"
        f"Long {case['long_label']}  /  Short {case['short_label']}\n"
        f"$100k notional  |  50bps borrow  |  S1=ConstStatic  S2=ConstDynamic  "
        f"S3=RollStatic  S4=RollDynamic",
        pad=12
    )

    # Panel 2 — drawdowns
    ax2 = axes[1]
    for sid in ('S1', 'S2', 'S3', 'S4'):
        m   = metrics[sid]
        lbl = f"{LABELS[sid]}  (max DD: {m['Max Drawdown %']:.1f}%)"
        ax2.plot(dates, tw[f'{sid}_Drawdown'],
                 color=COLORS[sid], linewidth=2, label=lbl)
    ax2.axhline(0, color='black', linewidth=0.8, alpha=0.4)
    ax2.yaxis.set_major_formatter(fmt)
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.set_title('Drawdown Comparison — All Strategies (%)',
                  fontweight='bold', pad=8)

    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout(h_pad=2)
    path = f"{OUTPUT_DIR}/{case_key}_3_all_strategies.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [3/4] Saved all-strategies chart → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — DYNAMIC POSITION SIGNALS (S2 and S4)
# ══════════════════════════════════════════════════════════════════════════════

def chart_dynamic_signals(tw, metrics, case, case_key):
    """
    Four-panel chart for the two dynamic strategies.
    Row 1: S2 return + S2 position signal.
    Row 2: S4 return + S4 position signal.
    """
    if tw is None:
        print(f"  [4/4] Skipped — no trade window data.")
        return

    dates  = tw['Date']
    entry  = pd.to_datetime(case['entry_date'])
    fmt    = pct_fmt()
    lt, st = case['long_ticker'], case['short_ticker']

    fig, axes = plt.subplots(4, 1, figsize=(16, 18),
                             gridspec_kw={'height_ratios': [2.5, 1, 2.5, 1]},
                             sharex=True)

    def _draw_pair(ax_r, ax_p, sid, beta_label):
        m = metrics[sid]
        # Return panel
        ax_r.plot(dates, tw['LONG_Cum_Ret'],  color='#aec7e8',
                  linewidth=1.2, linestyle=':', label=f'Long {lt} only')
        ax_r.plot(dates, tw['SHORT_Cum_Ret'], color='#ffbb78',
                  linewidth=1.2, linestyle=':', label=f'Short {st} only')
        ax_r.plot(dates, tw[f'{sid}_Cum_Ret'], color=COLORS[sid], linewidth=2.5,
                  label=(f"{LABELS[sid]}  ({m['Final Ret %']:.1f}%  "
                         f"Sharpe={m['Sharpe (Period)']:.2f}  "
                         f"MaxDD={m['Max Drawdown %']:.1f}%)"))
        ax_r.axhline(0, color='black', linestyle='--', alpha=0.4)
        ax_r.yaxis.set_major_formatter(fmt)
        ax_r.set_ylabel('Cumulative Return (%)')
        ax_r.legend(loc='upper left', fontsize=9)
        add_entry_vline(ax_r, entry)
        ax_r.set_title(
            f"{case['title']} — {LABELS[sid]}  |  {beta_label}\n"
            f"Days in market: {m['Days in Market']}  "
            f"({m['Long Days']} standard / {m['Short Days']} inverted)",
            pad=10
        )
        # Position signal panel
        pos = tw[f'{sid}_Position']
        ax_p.fill_between(dates, pos, 0, where=(pos > 0),
                          color='#2ca02c', alpha=0.55,
                          label=f'Standard: Long {lt} / Short {st}')
        ax_p.fill_between(dates, pos, 0, where=(pos < 0),
                          color='#d62728', alpha=0.55,
                          label=f'Inverted: Short {lt} / Long {st}')
        ax_p.axhline(0, color='black', linewidth=0.8, alpha=0.5)
        ax_p.set_yticks([-1, 0, 1])
        ax_p.set_yticklabels(['Inverted', 'Flat', 'Standard'], fontsize=9)
        ax_p.set_ylabel('Position')
        ax_p.legend(loc='upper right', fontsize=9)
        ax_p.set_title('Position Signal (lagged 1-day z-score trigger)',
                       fontweight='bold', pad=6)

    _draw_pair(axes[0], axes[1], 'S2',
               f"Constant β = {metrics['S2']['Const Beta']:.4f}")
    _draw_pair(axes[2], axes[3], 'S4',
               f"Rolling β — entry={metrics['S4']['Beta Entry']:.3f}  "
               f"mean={metrics['S4']['Beta Mean']:.3f}  "
               f"range={metrics['S4']['Beta Range']}")

    axes[3].xaxis.set_major_locator(mdates.MonthLocator())
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout(h_pad=2)
    path = f"{OUTPUT_DIR}/{case_key}_4_dynamic_signals.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [4/4] Saved dynamic signals     → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY CHART 1 — ALL STRATEGIES × ALL CASES (2×2 bar chart)
# ══════════════════════════════════════════════════════════════════════════════

def chart_summary_all_strategies(all_metrics):
    """
    2×2 grouped bar chart: Final Return, Sharpe, Max Drawdown, Win Rate.
    Each metric panel groups by case; within each group there are four bars
    (one per strategy), colour-coded by strategy ID.
    """
    fmt   = pct_fmt()
    cases = list(all_metrics.keys())
    sids  = ['S1', 'S2', 'S3', 'S4']
    x     = np.arange(len(cases))
    w     = 0.18
    offs  = np.array([-1.5, -0.5, 0.5, 1.5]) * w
    disp  = {'anadarko': 'Anadarko\n(2019)',
              'hulu':     'Hulu\n(2023)',
              'ksu':      'KSU\n(2021)'}
    xlbls = [disp.get(c, c) for c in cases]

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()

    specs = [
        (axes[0], 'Final Ret %',     'Cumulative Return (%)', fmt,  '{v:.1f}%'),
        (axes[1], 'Sharpe (Period)', 'Sharpe Ratio (Period)', None, '{v:.2f}'),
        (axes[2], 'Max Drawdown %',  'Max Drawdown (%)',      fmt,  '{v:.1f}%'),
        (axes[3], 'Win Rate',        'Win Rate (%)',          fmt,  '{v:.0f}%'),
    ]

    for ax, key, ylabel, formatter, lfmt in specs:
        for sid, off in zip(sids, offs):
            if key == 'Max Drawdown %':
                vals = [abs(all_metrics[c][sid][key]) for c in cases]
            else:
                vals = [all_metrics[c][sid][key] for c in cases]
            bars = ax.bar(x + off, vals, w, label=LABELS[sid],
                          color=COLORS[sid], alpha=0.85, edgecolor='white')
            for bar, v in zip(bars, vals):
                yoff = max(abs(bar.get_height()) * 0.04, 0.05)
                ypos = bar.get_height() + (yoff if bar.get_height() >= 0 else -yoff * 3)
                ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                        lfmt.format(v=v),
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(xlbls, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        if formatter:
            ax.yaxis.set_major_formatter(formatter)

    axes[0].set_title('Final Return (%)',              fontweight='bold', pad=10)
    axes[1].set_title('Sharpe Ratio (Period, √n)',     fontweight='bold', pad=10)
    axes[1].axhline(1, color='gray', linewidth=1, linestyle='--', alpha=0.6)
    axes[2].set_title('Max Drawdown (%, positive)',    fontweight='bold', pad=10)
    axes[3].set_title('Win Rate (%)',                  fontweight='bold', pad=10)

    fig.suptitle(
        "Winner's Curse Pairs Trade — All Four Strategies × All Cases\n"
        "S1: Const β / Static   S2: Const β / Dynamic   "
        "S3: Rolling β / Static   S4: Rolling β / Dynamic\n"
        "$100k notional  |  50bps borrow  |  4.5% risk-free",
        fontsize=12, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/summary_all_strategies.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved all-strategies summary    → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY CHART 2 — CONSTANT vs ROLLING BETA HEAD-TO-HEAD
# ══════════════════════════════════════════════════════════════════════════════

def chart_summary_beta_comparison(all_metrics):
    """
    2-row × 3-col grid. Row = entry style (static / dynamic). Col = case.
    Each cell: grouped bar showing final return for const-β vs rolling-β.
    Immediately answers: 'what did switching beta actually change?'
    """
    fmt   = pct_fmt()
    cases = list(all_metrics.keys())
    disp  = {'anadarko': 'Anadarko (2019)',
              'hulu':     'Hulu (2023)',
              'ksu':      'KSU (2021)'}

    fig, axes = plt.subplots(2, 3, figsize=(22, 10))

    row_specs = [
        ('Static entry',  'S1', 'S3', 0),
        ('Dynamic entry', 'S2', 'S4', 1),
    ]

    for row_label, const_sid, roll_sid, row in row_specs:
        for col, ck in enumerate(cases):
            ax   = axes[row][col]
            mc   = all_metrics[ck][const_sid]
            mr   = all_metrics[ck][roll_sid]
            vals = [mc['Final Ret %'], mr['Final Ret %']]
            bars = ax.bar([0, 1], vals,
                          color=[COLORS[const_sid], COLORS[roll_sid]],
                          alpha=0.85, width=0.45, edgecolor='white')
            for xp, (bar, v) in enumerate(zip(bars, vals)):
                yoff = abs(v) * 0.05 + 0.2
                ypos = v + (yoff if v >= 0 else -yoff * 2.5)
                ax.text(xp, ypos, f'{v:.2f}%',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
            ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
            ax.yaxis.set_major_formatter(fmt)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Const β', 'Rolling β'], fontsize=10)
            ax.set_title(f"{disp[ck]}\n{row_label}", pad=8)
            if col == 0:
                ax.set_ylabel('Final Return (%)')

            # Sharpe annotation
            ax.text(0, ax.get_ylim()[0] * 0.85,
                    f'Sharpe: {mc["Sharpe (Period)"]:.2f}',
                    ha='center', fontsize=8, color=COLORS[const_sid])
            ax.text(1, ax.get_ylim()[0] * 0.85,
                    f'Sharpe: {mr["Sharpe (Period)"]:.2f}',
                    ha='center', fontsize=8, color=COLORS[roll_sid])

    fig.suptitle(
        "Constant β vs Rolling β — Final Return by Entry Style and Case\n"
        "Top row: Static entry (hold from catalyst)  |  "
        "Bottom row: Dynamic entry (z-score ±2σ signal)",
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/summary_beta_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved beta comparison summary   → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    setup_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_metrics = {}

    for case_key, case in CASES.items():
        print(f"\n{'═'*64}")
        print(f"  {case['title']}")
        print(f"{'═'*64}")

        df = fetch_data(case['long_ticker'], case['short_ticker'],
                        case['fetch_start'],  case['fetch_end'])

        tw, metrics = run_all_backtests(df, case['entry_date'])

        if metrics:
            all_metrics[case_key] = metrics
            cw = 10
            div = '-' * (26 + cw * 5)
            print(f"\n  {'Strategy':<24} {'Return':>{cw}} {'Sharpe(P)':>{cw}} "
                  f"{'Ann.Sharpe':>{cw}} {'Max DD':>{cw}} {'Win%':>{cw}}")
            print(f"  {div}")
            for sid in ('S1', 'S2', 'S3', 'S4'):
                m = metrics[sid]
                print(f"  {LABELS[sid]:<24} "
                      f"{m['Final Ret %']:>{cw}.2f}%"
                      f"  {m['Sharpe (Period)']:>{cw}.3f}"
                      f"  {m['Sharpe (Annual)']:>{cw}.3f}"
                      f"  {m['Max Drawdown %']:>{cw}.2f}%"
                      f"  {m['Win Rate']:>{cw}.1f}%")
                if m['Entry Type'] == 'Dynamic':
                    print(f"  {'':24}   in-mkt={m['Days in Market']}d  "
                          f"({m['Long Days']} std / {m['Short Days']} inv)")
                if m['Beta Type'] == 'Constant':
                    print(f"  {'':24}   β={m['Const Beta']:.4f}  "
                          f"({m['Pre-Entry Days']} pre-entry days)")
                else:
                    print(f"  {'':24}   β entry={m['Beta Entry']:.4f}  "
                          f"mean={m['Beta Mean']:.4f}  range={m['Beta Range']}")
            print(f"  {div}")

        print(f"\n  Generating charts...")
        chart_normalized_returns(df, case, case_key)
        chart_spread_zscore(df, case, case_key)
        print(f"  Per-strategy charts:")
        chart_per_strategy(tw, metrics, case, case_key)
        chart_all_strategies(tw, metrics, case, case_key)
        chart_dynamic_signals(tw, metrics, case, case_key)

    # Summary charts
    if len(all_metrics) == len(CASES):
        print(f"\n{'═'*64}")
        print(f"  Generating summary charts...")
        chart_summary_all_strategies(all_metrics)
        chart_summary_beta_comparison(all_metrics)

    # Final cross-case print table
    print(f"\n{'═'*84}")
    print(f"  FINAL CROSS-CASE SUMMARY")
    print(f"{'═'*84}")
    print(f"  {'Case':<22} {'Strategy':<24} {'Return':>8} "
          f"{'Sharpe':>8} {'Max DD':>8} {'Win%':>7}")
    print(f"  {'-'*75}")
    disp_short = {'anadarko': 'Anadarko 2019',
                  'hulu':     'Hulu     2023',
                  'ksu':      'KSU      2021'}
    for ck in CASES:
        if ck not in all_metrics:
            continue
        for i, sid in enumerate(('S1', 'S2', 'S3', 'S4')):
            m   = all_metrics[ck][sid]
            lbl = disp_short[ck] if i == 0 else ''
            print(f"  {lbl:<22} {LABELS[sid]:<24} "
                  f"{m['Final Ret %']:>7.2f}%  "
                  f"{m['Sharpe (Period)']:>7.2f}  "
                  f"{m['Max Drawdown %']:>7.2f}%  "
                  f"{m['Win Rate']:>5.1f}%")
        print(f"  {'-'*75}")

    print(f"\n  All charts saved to '{OUTPUT_DIR}/'")
    print(f"\n  Per-case files (× 3 cases):")
    print(f"    <case>_1_normalized_returns.png")
    print(f"    <case>_2_spread_zscore.png          (const-β + rolling-β panels)")
    print(f"    <case>_S1_const_static.png           (S1 dedicated chart)")
    print(f"    <case>_S2_const_dynamic.png          (S2 dedicated chart + signal)")
    print(f"    <case>_S3_roll_static.png            (S3 dedicated chart)")
    print(f"    <case>_S4_roll_dynamic.png           (S4 dedicated chart + signal)")
    print(f"    <case>_3_all_strategies.png          (all 4 strategies overlaid)")
    print(f"    <case>_4_dynamic_signals.png         (S2 + S4 position signals)")
    print(f"\n  Summary files:")
    print(f"    summary_all_strategies.png           (2×2 bar: all strategies)")
    print(f"    summary_beta_comparison.png          (const vs rolling β)")


if __name__ == '__main__':
    main()