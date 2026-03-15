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
  ③ WWE (2023)      — FOXA vs EDR

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
    Beta-neutral pairs backtest.
    Returns trade_window DataFrame with all PnL columns populated,
    plus a metrics dict.
    """
    entry = pd.to_datetime(entry_date)
    df    = df.copy()

    # Hedge ratio on full history (lagged)
    df['Hedge_Ratio'] = compute_hedge_ratio(df)

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
                             - CAPITAL * tw['Hedge_Ratio'] * tw['SHORT_Ret']
                             - CAPITAL * tw['Hedge_Ratio'] * daily_borrow)

    for leg in ('LONG', 'SHORT', 'PAIR'):
        tw[f'{leg}_Cum_PnL'] = tw[f'{leg}_Daily_PnL'].cumsum()

    tw['PAIR_Peak']     = tw['PAIR_Cum_PnL'].cummax()
    tw['PAIR_Drawdown'] = tw['PAIR_Cum_PnL'] - tw['PAIR_Peak']

    def _sharpe(pnl, scale):
        s = pnl.iloc[1:]
        e = (s / CAPITAL) - (RISK_FREE / TRADING_DAYS)
        return (e.mean() / e.std()) * np.sqrt(scale) if e.std() != 0 else np.nan

    metrics = {
        'Final PnL':        tw['PAIR_Cum_PnL'].iloc[-1],
        'Max Drawdown':     (tw['PAIR_Cum_PnL'] - tw['PAIR_Cum_PnL'].cummax()).min(),
        'Sharpe (Period)':  _sharpe(tw['PAIR_Daily_PnL'], len(tw) - 1),
        'Sharpe (Annual)':  _sharpe(tw['PAIR_Daily_PnL'], TRADING_DAYS),
        'Win Rate':         (tw['PAIR_Daily_PnL'] > 0).mean() * 100,
        'Beta Entry':       tw['Hedge_Ratio'].iloc[0],
        'Beta Exit':        tw['Hedge_Ratio'].iloc[-1],
        'Beta Mean':        tw['Hedge_Ratio'].mean(),
        'Beta Min':         tw['Hedge_Ratio'].min(),
        'Beta Max':         tw['Hedge_Ratio'].max(),
        'Beta Std':         tw['Hedge_Ratio'].std(),
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
    Chart 3 of 3: 4-panel backtest chart.
      Panel 1 — Pairs strategy cumulative PnL with profit/loss shading
      Panel 2 — All three legs overlaid (long only, short only, pairs)
      Panel 3 — Drawdown
      Panel 4 — Rolling hedge ratio (β) over the trade window
    """
    if tw is None or metrics is None:
        print(f"  [3/3] Skipped — backtest returned no data.")
        return

    entry      = pd.to_datetime(case['entry_date'])
    dates      = tw['Date']
    pair_final = metrics['Final PnL']
    pair_sp    = metrics['Sharpe (Period)']
    pair_mdd   = metrics['Max Drawdown']

    fig, axes = plt.subplots(
        4, 1, figsize=(14, 18),
        gridspec_kw={'height_ratios': [3, 2, 1.5, 1.5]},
        sharex=True
    )

    # ── Panel 1: Pairs PnL ────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(dates, tw['PAIR_Cum_PnL'], color='#2ca02c', linewidth=2.5,
             label=f"Long {case['long_label']} / Short {case['short_label']} (β-Neutral)")
    ax1.axhline(0, color='black', linestyle='--', alpha=0.4)
    ax1.fill_between(dates, tw['PAIR_Cum_PnL'], 0,
                     where=(tw['PAIR_Cum_PnL'] >= 0),
                     color='#2ca02c', alpha=0.12, label='Profit Zone')
    ax1.fill_between(dates, tw['PAIR_Cum_PnL'], 0,
                     where=(tw['PAIR_Cum_PnL'] < 0),
                     color='#d62728', alpha=0.12, label='Loss Zone')
    ax1.set_title(
        f"{case['title']}\n"
        f"Pairs Trade Cumulative P&L (β-Neutral, after 50bps borrow cost)\n"
        f"Final PnL: ${pair_final:,.0f}  |  Sharpe (Period): {pair_sp:.2f}  "
        f"|  Max DD: ${pair_mdd:,.0f}",
        fontweight='bold', pad=12
    )
    ax1.set_ylabel('Cumulative P&L ($)')
    ax1.legend(loc='upper left', fontsize=10)

    trans = transforms.blended_transform_factory(ax1.transData, ax1.transAxes)
    ax1.axvline(entry, color='black', linestyle='-.', linewidth=1.5, alpha=0.7)
    ax1.text(entry, 0.95, '  Trade Entry', transform=trans,
             rotation=90, va='top', ha='right',
             fontsize=9, fontweight='bold', color='#333333')

    # ── Panel 2: Individual legs ──────────────────────────────────────
    ax2 = axes[1]
    long_final   = tw['LONG_Cum_PnL'].iloc[-1]
    short_final  = tw['SHORT_Cum_PnL'].iloc[-1]
    long_ticker  = case['long_ticker']
    short_ticker = case['short_ticker']

    ax2.plot(dates, tw['LONG_Cum_PnL'],  color='#1f77b4', linewidth=2,
             linestyle='--', label=f"Long {long_ticker} Only  (${long_final:,.0f})")
    ax2.plot(dates, tw['SHORT_Cum_PnL'], color='#ff7f0e', linewidth=2,
             linestyle='--', label=f"Short {short_ticker} Only (${short_final:,.0f})")
    ax2.plot(dates, tw['PAIR_Cum_PnL'],  color='#2ca02c', linewidth=2.5,
             label=f"Pairs Trade β-Neutral (${pair_final:,.0f})")
    ax2.axhline(0, color='black', linestyle='--', alpha=0.4)
    ax2.set_title('Individual Leg Benchmarks vs. Pairs Strategy',
                  fontweight='bold', pad=8)
    ax2.set_ylabel('Cumulative P&L ($)')
    ax2.legend(loc='upper left', fontsize=10)

    # ── Panel 3: Drawdown ─────────────────────────────────────────────
    ax3 = axes[2]
    ax3.fill_between(dates, tw['PAIR_Drawdown'], 0, color='#d62728', alpha=0.4)
    ax3.plot(dates, tw['PAIR_Drawdown'], color='#d62728', linewidth=1.5,
             label='Drawdown')
    ax3.axhline(pair_mdd, color='darkred', linestyle=':', linewidth=1.5,
                label=f'Max Drawdown: ${pair_mdd:,.0f}')
    ax3.set_title('Pairs Strategy Drawdown', fontweight='bold', pad=8)
    ax3.set_ylabel('Drawdown ($)')
    ax3.legend(loc='lower left', fontsize=10)

    # ── Panel 4: Rolling hedge ratio (β) ──────────────────────────────
    ax4 = axes[3]
    ax4.plot(dates, tw['Hedge_Ratio'], color='#9467bd', linewidth=2,
             label='Rolling β (20-day, lagged 1d)')
    ax4.axhline(metrics['Beta Mean'], color='#9467bd', linestyle='--',
                alpha=0.6, linewidth=1.2,
                label=f"Mean β = {metrics['Beta Mean']:.3f}")
    ax4.axhline(1.0, color='black', linestyle=':', alpha=0.3, linewidth=1)

    # Shade ±1 std band around mean
    beta_mean = metrics['Beta Mean']
    beta_std  = metrics['Beta Std']
    ax4.fill_between(dates,
                     beta_mean - beta_std,
                     beta_mean + beta_std,
                     color='#9467bd', alpha=0.08,
                     label=f'±1σ band  (σ = {beta_std:.3f})')

    # Annotate entry and exit values
    ax4.annotate(
        f"Entry β={metrics['Beta Entry']:.3f}",
        xy=(dates.iloc[0], metrics['Beta Entry']),
        xytext=(dates.iloc[min(5, len(dates)-1)], metrics['Beta Entry'] + beta_std * 0.6),
        fontsize=9, color='#9467bd', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#9467bd', lw=1.2),
    )
    ax4.annotate(
        f"Exit β={metrics['Beta Exit']:.3f}",
        xy=(dates.iloc[-1], metrics['Beta Exit']),
        xytext=(dates.iloc[max(-6, -len(dates))], metrics['Beta Exit'] + beta_std * 0.6),
        fontsize=9, color='#555555', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2),
    )

    ax4.set_title(
        f"Rolling Hedge Ratio β  "
        f"(Entry: {metrics['Beta Entry']:.3f}  |  "
        f"Exit: {metrics['Beta Exit']:.3f}  |  "
        f"Mean: {metrics['Beta Mean']:.3f}  |  "
        f"Range: {metrics['Beta Min']:.3f}–{metrics['Beta Max']:.3f})",
        fontweight='bold', pad=8
    )
    ax4.set_ylabel('Hedge Ratio (β)')
    ax4.legend(loc='upper left', fontsize=9)

    ax4.xaxis.set_major_locator(mdates.MonthLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout(h_pad=2.5)
    path = f"{OUTPUT_DIR}/{case_key}_3_backtest_pnl.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [3/3] Saved backtest PnL        → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-CASE SUMMARY CHART
# ══════════════════════════════════════════════════════════════════════════════

def chart_cross_case_summary(all_metrics):
    """
    Bar chart comparing Final PnL and Sharpe (Period) across all three
    real-world cases. Intended as a single-slide summary for the presentation.
    """
    labels = list(all_metrics.keys())
    pnls   = [all_metrics[k]['Final PnL']       for k in labels]
    sharps = [all_metrics[k]['Sharpe (Period)']  for k in labels]
    mdds   = [abs(all_metrics[k]['Max Drawdown']) for k in labels]

    display_labels = {
        'anadarko': 'Anadarko\nLong CVX / Short OXY\n(2019)',
        'hulu':     'Hulu\nLong CMCSA / Short DIS\n(2023)',
        'wwe':      'WWE\nLong FOXA / Short EDR\n(2023)',
        'ksu':      'KSU Railway\nLong CNI / Short CP\n(2021)',
    }
    x_labels = [display_labels.get(k, k) for k in labels]

    bar_colors = ['#2ca02c' if p >= 0 else '#d62728' for p in pnls]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Panel A — Final PnL
    ax1 = axes[0]
    bars = ax1.bar(x_labels, pnls, color=bar_colors, edgecolor='white',
                   linewidth=1.5, width=0.5)
    ax1.axhline(0, color='black', linewidth=1, alpha=0.5)
    for bar, val in zip(bars, pnls):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (max(pnls) * 0.02),
                 f'${val:,.0f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_title('Final PnL\n($100k notional, 50bps borrow)',
                  fontweight='bold', pad=10)
    ax1.set_ylabel('Cumulative P&L ($)')
    ax1.tick_params(axis='x', labelsize=10)

    # Panel B — Sharpe (Period)
    ax2 = axes[1]
    sharpe_colors = ['#2ca02c' if s >= 0 else '#d62728' for s in sharps]
    bars2 = ax2.bar(x_labels, sharps, color=sharpe_colors, edgecolor='white',
                    linewidth=1.5, width=0.5)
    ax2.axhline(0, color='black', linewidth=1, alpha=0.5)
    ax2.axhline(1, color='gray', linewidth=1, linestyle='--', alpha=0.6,
                label='Sharpe = 1.0')
    for bar, val in zip(bars2, sharps):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.03,
                 f'{val:.2f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_title('Sharpe Ratio (Period)\nscaled by √n actual days',
                  fontweight='bold', pad=10)
    ax2.set_ylabel('Sharpe Ratio')
    ax2.legend(fontsize=10)
    ax2.tick_params(axis='x', labelsize=10)

    # Panel C — Max Drawdown (shown as positive for readability)
    ax3 = axes[2]
    bars3 = ax3.bar(x_labels, mdds, color='#ff7f0e', edgecolor='white',
                    linewidth=1.5, width=0.5)
    for bar, val in zip(bars3, mdds):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (max(mdds) * 0.02),
                 f'${val:,.0f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax3.set_title('Max Drawdown\n(shown as positive)',
                  fontweight='bold', pad=10)
    ax3.set_ylabel('Max Drawdown ($)')
    ax3.tick_params(axis='x', labelsize=10)

    fig.suptitle(
        "Winner's Curse Pairs Trade — Out-of-Sample Validation Summary\n"
        "Long clean-balance-sheet bidder / Short levered winner  (β-Neutral, $100k notional)",
        fontsize=14, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/summary_cross_case_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved cross-case summary chart → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    setup_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_metrics = {}

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

        # 2. Run backtest
        tw, metrics = run_backtest(df, case['entry_date'])

        if metrics:
            all_metrics[case_key] = metrics
            print(f"\n  Pairs Trade results:")
            print(f"    Final PnL:       ${metrics['Final PnL']:>10,.2f}")
            print(f"    Sharpe (Period):  {metrics['Sharpe (Period)']:>10.3f}")
            print(f"    Sharpe (Annual):  {metrics['Sharpe (Annual)']:>10.3f}")
            print(f"    Max Drawdown:    ${metrics['Max Drawdown']:>10,.2f}")
            print(f"    Win Rate:         {metrics['Win Rate']:>9.1f}%")
            print(f"\n  Hedge Ratio (β) over trade window:")
            print(f"    Entry β:          {metrics['Beta Entry']:>10.4f}")
            print(f"    Exit β:           {metrics['Beta Exit']:>10.4f}")
            print(f"    Mean β:           {metrics['Beta Mean']:>10.4f}")
            print(f"    Min β:            {metrics['Beta Min']:>10.4f}")
            print(f"    Max β:            {metrics['Beta Max']:>10.4f}")
            print(f"    Std β:            {metrics['Beta Std']:>10.4f}")

        # 3. Generate the three charts
        print(f"\n  Generating charts...")
        chart_normalized_returns(df, case, case_key)
        chart_spread_zscore(df, case, case_key)
        chart_backtest_pnl(df, tw, metrics, case, case_key)

    # 4. Cross-case summary (only if we have results for all cases)
    if len(all_metrics) == len(CASES):
        chart_cross_case_summary(all_metrics)

    # 5. Final summary table
    print(f"\n{'═'*60}")
    print(f"  FINAL CROSS-CASE SUMMARY (β-Neutral Pairs Trade)")
    print(f"{'═'*60}")
    print(f"  {'Case':<44} {'PnL':>10} {'Sharpe(P)':>10} {'Max DD':>10} {'Win%':>8}")
    print(f"  {'-'*82}")
    labels_display = {
        'anadarko': 'Anadarko 2019 — Long CVX  / Short OXY ',
        'hulu':     'Hulu     2023 — Long CMCSA / Short DIS ',
        'wwe':      'WWE      2023 — Long FOXA  / Short EDR ',
        'ksu':      'KSU      2021 — Long CNI   / Short CP  ',
    }
    for k, lbl in labels_display.items():
        if k in all_metrics:
            m = all_metrics[k]
            print(f"  {lbl:<44} "
                  f"${m['Final PnL']:>8,.0f}  "
                  f"{m['Sharpe (Period)']:>9.2f}  "
                  f"${m['Max Drawdown']:>8,.0f}  "
                  f"{m['Win Rate']:>6.1f}%")

    print(f"\n  All charts saved to '{OUTPUT_DIR}/'")
    print(f"  Files generated per case:")
    print(f"    <case>_1_normalized_returns.png")
    print(f"    <case>_2_spread_zscore.png")
    print(f"    <case>_3_backtest_pnl.png")
    print(f"    summary_cross_case_comparison.png")


if __name__ == '__main__':
    main()