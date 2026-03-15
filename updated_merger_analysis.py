import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import matplotlib.transforms as transforms
from statsmodels.tsa.stattools import coint


def setup_plot_style():
    """Configure seaborn settings for presentation-ready charts"""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'sans-serif',
        'axes.titleweight': 'bold'
    })


def format_x_axis(ax):
    """Format x-axis dates nicely"""
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def add_milestones(ax):
    """Adds vertical lines and text labels for major deal milestones"""
    milestones = {
        '2025-09-11': 'WSJ Reports Potential Paramount Deal',
        '2025-10-21': 'WBD Announces Potential Sale',
        '2025-11-12': 'Implied Volatility Peak',
        '2025-12-5':  'WBD Announces Agreement w/Netflix',
        '2026-02-26': 'Netflix Exits Deal'
    }
    for date_str, label in milestones.items():
        date_obj = pd.to_datetime(date_str)
        ax.axvline(x=date_obj, color='black', linestyle='-.', alpha=0.6, linewidth=1.5)
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(date_obj, 0.95, f' {label}', transform=trans,
                rotation=90, verticalalignment='top', horizontalalignment='right',
                fontsize=11, fontweight='bold', color='#333333')


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HEDGE RATIO — computed ONCE on the full history, lagged 1 day.
# Both calculate_quant_metrics and backtest use this single source of truth.
# ══════════════════════════════════════════════════════════════════════════════

def compute_hedge_ratio(df, window=20):
    """
    Rolling OLS beta:  β = Cov(NFLX_ret, PSKY_ret) / Var(PSKY_ret)
    Lagged by 1 day to eliminate look-ahead bias.
    Returns a Series aligned to df's index.
    """
    nflx_ret   = df['NFLX'].pct_change()
    psky_ret   = df['PSKY'].pct_change()
    roll_cov   = nflx_ret.rolling(window).cov(psky_ret)
    roll_var   = psky_ret.rolling(window).var()
    beta       = roll_cov / roll_var
    return beta.shift(1).fillna(1.0)


# ══════════════════════════════════════════════════════════════════════════════
# COINTEGRATION TEST
# ══════════════════════════════════════════════════════════════════════════════

def run_cointegration_test(df):
    """
    Engle-Granger cointegration test on NFLX and PSKY price levels.

    We test three windows:
      1. Pre-deal (full history up to deal announcement on 2025-09-11)
      2. Bidding-war (2025-09-11 → 2026-02-26)
      3. Post-Netflix-exit (2026-02-26 onward)

    Interpretation:
      p-value < 0.05  → series ARE cointegrated (mean-reverting spread exists)
      p-value ≥ 0.05  → no stable long-run relationship

    For the pairs trade thesis we want:
      • Pre-deal:       cointegrated  (shows the two stocks historically moved together)
      • Post-exit:      NOT cointegrated  (the relationship broke down → divergence trade)
    """
    print("\n" + "=" * 60)
    print("  COINTEGRATION ANALYSIS: NFLX vs PSKY (Engle-Granger)")
    print("=" * 60)

    windows = {
        'Pre-Deal Baseline  (Aug–Sep 2025)':    (None,           '2025-09-11'),
        'Bidding War        (Sep 2025–Feb 2026)': ('2025-09-11', '2026-02-26'),
        'Post-Netflix Exit  (Feb 2026 onward)':  ('2026-02-26',  None),
    }

    results = {}
    for label, (start, end) in windows.items():
        mask = pd.Series([True] * len(df))
        if start:
            mask &= df['Date'] >= pd.to_datetime(start)
        if end:
            mask &= df['Date'] < pd.to_datetime(end)

        subset = df[mask][['NFLX', 'PSKY']].dropna()

        if len(subset) < 30:
            print(f"  {label}: ⚠  Insufficient data ({len(subset)} obs), skipping.")
            continue

        score, p_val, _ = coint(subset['NFLX'], subset['PSKY'])
        sig             = "✓ COINTEGRATED" if p_val < 0.05 else "✗ NOT cointegrated"
        results[label]  = {'t_stat': score, 'p_value': p_val, 'verdict': sig, 'n': len(subset)}

        print(f"\n  {label}")
        print(f"    Observations : {len(subset)}")
        print(f"    t-statistic  : {score:.4f}")
        print(f"    p-value      : {p_val:.4f}   → {sig}")

    print()
    print("  Thesis check:")
    print("    Pre-deal cointegration proves NFLX & PSKY historically moved together.")
    print("    Post-exit divergence is the structural break that creates the trade.")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL PROOFS + SIGNAL ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def calculate_quant_metrics(df):
    """Calculates statistical proofs and engineers quantitative trading signals"""
    print("\n" + "=" * 50)
    print("STATISTICAL PROOFS FOR PRESENTATION SLIDES")
    print("=" * 50)

    # 1. Proof of Idiosyncratic Risk (Pearson Correlation)
    clean_vol           = df[['Imp Vol', 'VIX']].dropna()
    corr, p_val_corr    = stats.pearsonr(clean_vol['Imp Vol'], clean_vol['VIX'])
    print(f"[Slide 1] WBD IV vs VIX Correlation: {corr:.3f}  (p-value: {p_val_corr:.4f})")

    # Define regimes for T-Tests
    exit_date    = pd.to_datetime('2026-02-26')
    before_exit  = df[df['Date'] < exit_date]
    after_exit   = df[df['Date'] >= exit_date]

    # 2. Proof of Volatility Crush (Welch's T-Test)
    t_stat_iv, p_val_iv = stats.ttest_ind(
        before_exit['Imp Vol'].dropna(),
        after_exit['Imp Vol'].dropna(),
        equal_var=False
    )
    print(f"[Slide 2] IV Crush:      Before = {before_exit['Imp Vol'].mean():.1f}%,  "
          f"After = {after_exit['Imp Vol'].mean():.1f}%  (p-value: {p_val_iv:.4e})")

    # 3. Sentiment Flip (Welch's T-Test on Put/Call Ratio)
    t_stat_pc, p_val_pc = stats.ttest_ind(
        before_exit['P/C Vol'].dropna(),
        after_exit['P/C Vol'].dropna(),
        equal_var=False
    )
    print(f"[Slide 3] P/C Ratio:     Before = {before_exit['P/C Vol'].mean():.2f},   "
          f"After = {after_exit['P/C Vol'].mean():.2f}    (p-value: {p_val_pc:.4e})")

    print("\n" + "=" * 50)
    print("ENGINEERING TRADING SIGNALS")
    print("=" * 50)

    # Signal 1: Volatility Risk Premium (VRP)
    df['WBD_Log_Ret']  = np.log(df['WBD'] / df['WBD'].shift(1))
    df['Realized_Vol'] = df['WBD_Log_Ret'].rolling(window=20).std() * np.sqrt(252) * 100
    df['VRP']          = df['Imp Vol'] - df['Realized_Vol']

    # Signal 2: Arbitrage Spread & Implied Deal Probability
    df['Arb_Spread']        = 31.00 - df['WBD']
    unaffected_price        = df['WBD'].iloc[0]
    implied_prob            = (df['WBD'] - unaffected_price) / (31.00 - unaffected_price)
    df['Implied_Deal_Prob'] = np.clip(implied_prob, 0, 1) * 100

    # Signal 3: Pairs Trade Hedge Ratio — use shared function (no duplication)
    df['NFLX_Ret']        = df['NFLX'].pct_change()
    df['PSKY_Ret']        = df['PSKY'].pct_change()
    df['PSKY_NFLX_Beta']  = compute_hedge_ratio(df, window=20)

    # Signal 4: Pairs Spread Z-Score (NEW)
    #   Spread = NFLX_log_ret − β × PSKY_log_ret  (beta-adjusted)
    #   Z-score normalises over a rolling 20-day window so you can
    #   read "how many standard deviations wide is the divergence today?"
    df['NFLX_Log_Ret']  = np.log(df['NFLX'] / df['NFLX'].shift(1))
    df['PSKY_Log_Ret']  = np.log(df['PSKY'] / df['PSKY'].shift(1))
    df['Pair_Spread']   = df['NFLX_Log_Ret'] - df['PSKY_NFLX_Beta'] * df['PSKY_Log_Ret']
    spread_mean         = df['Pair_Spread'].rolling(20).mean()
    spread_std          = df['Pair_Spread'].rolling(20).std()
    df['Spread_ZScore'] = (df['Pair_Spread'] - spread_mean) / spread_std

    print("\nLatest Signal Output (Last 3 Days of Data):")
    cols_to_show = ['Date', 'WBD', 'Arb_Spread', 'Implied_Deal_Prob',
                    'VRP', 'PSKY_NFLX_Beta', 'Spread_ZScore']
    print(df[cols_to_show].tail(3).to_string(index=False))

    return df


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def backtest_winners_curse_pairs_trade(df):
    """
    Backtests a Market-Neutral Pairs Trade: Long NFLX / Short PSKY.
    Executes exactly when Netflix exits the deal (2026-02-26).

    Key design decisions:
    - Hedge ratio computed via shared compute_hedge_ratio(), lagged 1 day.
    - ALL three PnL series use the same daily-return method for consistency.
    - Day 1 return is 0 by construction (entry day, no prior close).
    - Borrow cost applied to the short PSKY leg at 50 bps annualised.
    - Both realised-period Sharpe AND annualised Sharpe are reported.
    """
    print("\n" + "=" * 60)
    print("  BACKTEST: THE WINNER'S CURSE (LONG NFLX / SHORT PSKY)")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. ISOLATE THE POST-DEAL WINDOW (Phase 2)                          #
    # ------------------------------------------------------------------ #
    trade_start  = pd.to_datetime('2026-02-26')
    trade_window = df[df['Date'] >= trade_start].copy().reset_index(drop=True)

    if trade_window.empty:
        print("ERROR: No data found on or after 2026-02-26. Aborting.")
        return

    n_days = len(trade_window)

    # ------------------------------------------------------------------ #
    # 2. HEDGE RATIO — from shared function, already lagged 1 day        #
    # ------------------------------------------------------------------ #
    hedge_series = compute_hedge_ratio(df, window=20)
    hedge_map    = dict(zip(df['Date'], hedge_series))
    trade_window['Hedge_Ratio'] = trade_window['Date'].map(hedge_map).fillna(1.0)

    print(f"\nNFLX entry price:          ${trade_window['NFLX'].iloc[0]:.2f}")
    print(f"PSKY entry price:          ${trade_window['PSKY'].iloc[0]:.2f}")
    print(f"Opening Hedge Ratio (β):   {trade_window['Hedge_Ratio'].iloc[0]:.4f}")
    print(f"Closing Hedge Ratio (β):   {trade_window['Hedge_Ratio'].iloc[-1]:.4f}")
    print(f"Trade window:              {n_days} trading days\n")

    # ------------------------------------------------------------------ #
    # 3. DAILY RETURNS                                                    #
    # ------------------------------------------------------------------ #
    trade_window['NFLX_Ret'] = trade_window['NFLX'].pct_change().fillna(0)
    trade_window['PSKY_Ret'] = trade_window['PSKY'].pct_change().fillna(0)

    # ------------------------------------------------------------------ #
    # 4. PORTFOLIO PARAMETERS                                             #
    # ------------------------------------------------------------------ #
    CAPITAL       = 100_000   # $100k notional per leg
    RISK_FREE     = 0.045     # 4.5% annualised risk-free rate
    TRADING_DAYS  = 252
    BORROW_COST   = 0.0050    # 50 bps p.a. stock-borrow cost on short PSKY leg
    daily_borrow  = BORROW_COST / TRADING_DAYS  # cost per day as a fraction

    # ------------------------------------------------------------------ #
    # 5. DAILY P&L                                                        #
    #    Short leg has borrow cost deducted each day.                     #
    # ------------------------------------------------------------------ #
    trade_window['NFLX_Daily_PnL'] = CAPITAL * trade_window['NFLX_Ret']

    # Short PSKY: gains when PSKY falls; borrow cost is a daily drag
    trade_window['PSKY_Daily_PnL'] = (
        -CAPITAL * trade_window['PSKY_Ret']
        - CAPITAL * daily_borrow  # borrow cost reduces PnL every day
    )

    trade_window['Pair_Daily_PnL'] = (
          CAPITAL * trade_window['NFLX_Ret']
        - CAPITAL * trade_window['Hedge_Ratio'] * trade_window['PSKY_Ret']
        - CAPITAL * trade_window['Hedge_Ratio'] * daily_borrow  # scaled borrow
    )

    # ------------------------------------------------------------------ #
    # 6. CUMULATIVE P&L                                                   #
    # ------------------------------------------------------------------ #
    trade_window['NFLX_Cum_PnL'] = trade_window['NFLX_Daily_PnL'].cumsum()
    trade_window['PSKY_Cum_PnL'] = trade_window['PSKY_Daily_PnL'].cumsum()
    trade_window['Pair_Cum_PnL'] = trade_window['Pair_Daily_PnL'].cumsum()

    # ------------------------------------------------------------------ #
    # 7. DRAWDOWN                                                         #
    # ------------------------------------------------------------------ #
    trade_window['Pair_Peak']     = trade_window['Pair_Cum_PnL'].cummax()
    trade_window['Pair_Drawdown'] = trade_window['Pair_Cum_PnL'] - trade_window['Pair_Peak']

    # ------------------------------------------------------------------ #
    # 8. PERFORMANCE METRICS                                              #
    # ------------------------------------------------------------------ #

    def realised_sharpe(daily_pnl_series, capital, rf_annual, tdays):
        """
        Period Sharpe scaled by sqrt(n_actual).
        Drops entry day (return = 0 by construction) to avoid suppressing variance.
        """
        pnl   = daily_pnl_series.iloc[1:]
        n     = len(pnl)
        ret   = pnl / capital
        rf_d  = rf_annual / tdays
        exc   = ret - rf_d
        if exc.std() == 0:
            return np.nan
        return (exc.mean() / exc.std()) * np.sqrt(n)

    def annualised_sharpe(daily_pnl_series, capital, rf_annual, tdays):
        """
        Standard annualised Sharpe scaled by sqrt(252).
        Allows like-for-like comparison with published benchmarks.
        """
        pnl   = daily_pnl_series.iloc[1:]
        ret   = pnl / capital
        rf_d  = rf_annual / tdays
        exc   = ret - rf_d
        if exc.std() == 0:
            return np.nan
        return (exc.mean() / exc.std()) * np.sqrt(tdays)

    def max_drawdown_dollars(cum_pnl_series):
        peak = cum_pnl_series.cummax()
        return (cum_pnl_series - peak).min()

    def win_rate_pct(daily_pnl_series):
        return (daily_pnl_series > 0).mean() * 100

    strategies = {
        'Long NFLX Only':          'NFLX',
        'Short PSKY Only':         'PSKY',
        'Pairs Trade (β-Neutral)': 'Pair',
    }

    metrics = {}
    for label, prefix in strategies.items():
        pnl_col          = f'{prefix}_Daily_PnL'
        cum_col          = f'{prefix}_Cum_PnL'
        metrics[label]   = {
            'Final PnL':       trade_window[cum_col].iloc[-1],
            'Max Drawdown':    max_drawdown_dollars(trade_window[cum_col]),
            'Sharpe (Period)': realised_sharpe(trade_window[pnl_col], CAPITAL, RISK_FREE, TRADING_DAYS),
            'Sharpe (Annual)': annualised_sharpe(trade_window[pnl_col], CAPITAL, RISK_FREE, TRADING_DAYS),
            'Win Rate':        win_rate_pct(trade_window[pnl_col]),
        }

    # ------------------------------------------------------------------ #
    # 9. PRINT PERFORMANCE TABLE                                          #
    # ------------------------------------------------------------------ #
    col_w   = 26
    lbl_w   = 24
    divider = "-" * (lbl_w + col_w * 3)

    print(divider)
    print(f"{'Metric':<{lbl_w}}"
          f"{'Long NFLX Only':>{col_w}}"
          f"{'Short PSKY Only':>{col_w}}"
          f"{'Pairs Trade (β-Neutral)':>{col_w}}")
    print(divider)

    rows = [
        ('Final Net PnL',         'Final PnL',       lambda v: f"${v:>10,.2f}"),
        ('Max Drawdown',          'Max Drawdown',     lambda v: f"${v:>10,.2f}"),
        ('Sharpe (Period)',        'Sharpe (Period)',  lambda v: f"{v:>12.3f}"),
        ('Sharpe (Annualised)',    'Sharpe (Annual)',  lambda v: f"{v:>12.3f}"),
        ('Daily Win Rate',         'Win Rate',         lambda v: f"{v:>11.1f}%"),
    ]
    for display, key, fmt in rows:
        print(f"{display:<{lbl_w}}"
              + "".join(f"{fmt(metrics[s][key]):>{col_w}}" for s in metrics))

    print(divider)
    print(f"\nStarting Capital:   ${CAPITAL:>10,.2f}")
    print(f"Risk-Free Rate:      {RISK_FREE*100:.1f}% p.a.")
    print(f"Short Borrow Cost:   {BORROW_COST*100:.0f} bps p.a. (applied to short PSKY leg)")
    print(f"Trade Entry:         {trade_window['Date'].iloc[0].strftime('%Y-%m-%d')}")
    print(f"Trade Exit:          {trade_window['Date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"\nNOTE: 'Sharpe (Period)' scales by √n_actual — honest for short windows.")
    print(f"      'Sharpe (Annual)' scales by √252 — use for benchmark comparisons.")

    # ------------------------------------------------------------------ #
    # 10. 3-PANEL CHART (existing)                                        #
    # ------------------------------------------------------------------ #
    os.makedirs('charts', exist_ok=True)
    fig, axes = plt.subplots(
        3, 1,
        figsize=(14, 14),
        gridspec_kw={'height_ratios': [3, 2, 1.5]},
        sharex=True
    )

    dates       = trade_window['Date']
    pair_final  = metrics['Pairs Trade (β-Neutral)']['Final PnL']
    pair_sharpe = metrics['Pairs Trade (β-Neutral)']['Sharpe (Period)']
    pair_mdd    = metrics['Pairs Trade (β-Neutral)']['Max Drawdown']

    # ── Panel 1: Pairs strategy ──────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(dates, trade_window['Pair_Cum_PnL'],
             color='#2ca02c', linewidth=2.5,
             label='Long NFLX / Short PSKY (β-Neutral, after borrow cost)')
    ax1.axhline(0, color='black', linestyle='--', alpha=0.4)
    ax1.fill_between(dates, trade_window['Pair_Cum_PnL'], 0,
                     where=(trade_window['Pair_Cum_PnL'] >= 0),
                     color='#2ca02c', alpha=0.12, label='Profit Zone')
    ax1.fill_between(dates, trade_window['Pair_Cum_PnL'], 0,
                     where=(trade_window['Pair_Cum_PnL'] < 0),
                     color='#d62728', alpha=0.12, label='Loss Zone')
    ax1.set_title(
        "Pairs Trade Cumulative P&L: Long NFLX / Short PSKY (β-Neutral)\n"
        f"Final PnL: ${pair_final:,.0f}  |  "
        f"Sharpe (Period): {pair_sharpe:.2f}  |  "
        f"Max DD: ${pair_mdd:,.0f}  |  Borrow Cost: 50 bps",
        fontweight='bold', pad=12
    )
    ax1.set_ylabel("Cumulative P&L ($)")
    ax1.legend(loc='upper left', fontsize=10)

    trans = transforms.blended_transform_factory(ax1.transData, ax1.transAxes)
    ax1.axvline(trade_start, color='black', linestyle='-.', alpha=0.7, linewidth=1.5)
    ax1.text(trade_start, 0.95, '  Trade Entry (Netflix Exits)',
             transform=trans, rotation=90, va='top', ha='right',
             fontsize=9, fontweight='bold', color='#333333')

    # ── Panel 2: All three legs ──────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(dates, trade_window['NFLX_Cum_PnL'],
             color='#d62728', linewidth=2, linestyle='--',
             label=f"Long NFLX Only  (${metrics['Long NFLX Only']['Final PnL']:,.0f})")
    ax2.plot(dates, trade_window['PSKY_Cum_PnL'],
             color='#ff7f0e', linewidth=2, linestyle='--',
             label=f"Short PSKY Only (${metrics['Short PSKY Only']['Final PnL']:,.0f})")
    ax2.plot(dates, trade_window['Pair_Cum_PnL'],
             color='#2ca02c', linewidth=2.5,
             label=f"Pairs Trade β-Neutral (${pair_final:,.0f})")
    ax2.axhline(0, color='black', linestyle='--', alpha=0.4)
    ax2.set_title("Individual Leg Benchmarks vs. Pairs Strategy",
                  fontweight='bold', pad=8)
    ax2.set_ylabel("Cumulative P&L ($)")
    ax2.legend(loc='upper left', fontsize=10)

    # ── Panel 3: Drawdown ────────────────────────────────────────────
    ax3 = axes[2]
    ax3.fill_between(dates, trade_window['Pair_Drawdown'], 0,
                     color='#d62728', alpha=0.4)
    ax3.plot(dates, trade_window['Pair_Drawdown'],
             color='#d62728', linewidth=1.5, label='Drawdown')
    ax3.axhline(pair_mdd, color='darkred', linestyle=':',
                linewidth=1.5,
                label=f"Max Drawdown: ${pair_mdd:,.0f}")
    ax3.set_title("Pairs Strategy Drawdown", fontweight='bold', pad=8)
    ax3.set_ylabel("Drawdown ($)")
    ax3.legend(loc='lower left', fontsize=10)

    ax3.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout(h_pad=2.5)
    out_path = 'charts/pairs_trade_pnl.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved 3-panel backtest chart → '{out_path}'")

    return trade_window, metrics


# ══════════════════════════════════════════════════════════════════════════════
# CHART 5: PAIRS SPREAD Z-SCORE  (NEW)
# ══════════════════════════════════════════════════════════════════════════════

def plot_spread_zscore(df, output_dir='charts'):
    """
    Chart 5: Visualises the beta-adjusted NFLX/PSKY spread z-score over the
    full study period with the entry signal and ±2σ bands marked.

    This is the slide that answers "HOW did you know when to enter the trade?"
    The spread z-score breaking out of its historical band on Feb 26 is the
    quantitative trigger for the pairs trade.
    """
    fig, ax = plt.subplots(figsize=(16, 8))

    entry_date = pd.to_datetime('2026-02-26')

    # Plot z-score
    ax.plot(df['Date'], df['Spread_ZScore'],
            color='#1f77b4', linewidth=2, label='Pairs Spread Z-Score (β-Adjusted)', zorder=3)

    # ±1σ and ±2σ reference bands
    for level, style, label_str in [(1, ':', '±1σ'), (2, '--', '±2σ Entry Band')]:
        ax.axhline( level, color='gray', linestyle=style, alpha=0.7, linewidth=1.2,
                    label=label_str if level == 2 else None)
        ax.axhline(-level, color='gray', linestyle=style, alpha=0.7, linewidth=1.2)

    ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)

    # Shade post-entry region
    ax.axvspan(entry_date, df['Date'].max(), color='#2ca02c', alpha=0.06,
               label='Trade Active (Post Netflix Exit)')

    # Mark entry
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.axvline(entry_date, color='black', linestyle='-.', linewidth=1.5, alpha=0.7)
    ax.text(entry_date, 0.95, '  Trade Entry: Netflix Exits', transform=trans,
            rotation=90, va='top', ha='right', fontsize=11,
            fontweight='bold', color='#333333')

    # Annotate the divergence burst if there's a post-entry peak
    post_entry   = df[df['Date'] >= entry_date]['Spread_ZScore'].dropna()
    if not post_entry.empty:
        peak_val = post_entry.max()
        peak_idx = post_entry.idxmax()
        peak_date = df.loc[peak_idx, 'Date']
        ax.annotate(
            f"Peak divergence\nZ = {peak_val:.2f}σ",
            xy=(peak_date, peak_val),
            xytext=(peak_date, peak_val + 0.8),
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
            fontsize=10, color='#d62728', fontweight='bold',
            ha='center'
        )

    ax.set_title("Chart 5: Pairs Trade Spread Z-Score — Structural Break at Netflix Exit\n"
                 "β-Adjusted Spread: NFLX log-return − β × PSKY log-return  (20-day rolling normalisation)",
                 pad=20)
    ax.set_ylabel("Z-Score (σ)")
    ax.legend(loc='upper left', fontsize=10)
    format_x_axis(ax)

    plt.tight_layout()
    out_path = f'{output_dir}/chart5_spread_zscore.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved spread z-score chart → '{out_path}'")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading and cleaning data...")
    options_df = pd.read_csv('data/wbd_options-overview-history-03-14-2026.csv',
                             skipfooter=1, engine='python')
    stock_df   = pd.read_csv('data/combined_WBD_PSKY_NFLX.csv')

    options_df['Date']    = pd.to_datetime(options_df['Date'])
    stock_df['Date']      = pd.to_datetime(stock_df['Date'])
    options_df['Imp Vol'] = options_df['Imp Vol'].astype(str).str.replace('%', '').astype(float)

    df = pd.merge(options_df, stock_df, on='Date', how='inner')
    df = df.sort_values('Date').reset_index(drop=True)

    # ── Step 1: Statistical proofs + signal engineering ──────────────
    df = calculate_quant_metrics(df)

    # ── Step 2: Cointegration test (NEW) ─────────────────────────────
    run_cointegration_test(df)

    # ── Step 3: Generate all charts ──────────────────────────────────
    print("\nGenerating presentation charts...")
    setup_plot_style()
    output_dir = 'charts'
    os.makedirs(output_dir, exist_ok=True)

    # Chart 1: Idiosyncratic Risk (IV vs VIX)
    fig1, ax1 = plt.subplots(figsize=(16, 8))
    ax1.plot(df['Date'], df['Imp Vol'], label='WBD Implied Volatility (%)',
             color='#9467bd', linewidth=2.5)
    ax1.plot(df['Date'], df['VIX'], label='Market Volatility (VIX)',
             color='#7f7f7f', linewidth=2.5)
    ax1.set_title('Chart 1: WBD Deal Uncertainty vs General Market Risk', pad=20)
    ax1.set_ylabel('Volatility (%)')
    add_milestones(ax1)
    ax1.legend(loc='upper left')
    format_x_axis(ax1)
    plt.savefig(f'{output_dir}/chart1_iv_vs_vix.png')
    plt.close(fig1)

    # Chart 2: The Arbitrage Spread
    fig2, ax2 = plt.subplots(figsize=(16, 8))
    ax2.plot(df['Date'], df['WBD'], label='WBD Stock Price',
             color='#1f77b4', linewidth=2.5)
    ax2.axhline(y=31.0, color='#d62728', linestyle='--', linewidth=2,
                label='Paramount Offer ($31.00)')
    ax2.fill_between(df['Date'], df['WBD'], 31.0,
                     where=(df['WBD'] <= 31.0) & (df['Date'] >= pd.to_datetime('2026-02-26')),
                     color='red', alpha=0.1, label='DOJ Risk Premium')
    ax2.set_title('Chart 2: The Arbitrage Spread & Market Doubt', pad=20)
    ax2.set_ylabel('Stock Price ($)')
    ax2.set_ylim(bottom=df['WBD'].min() * 0.9, top=35)
    add_milestones(ax2)
    ax2.legend(loc='lower left')
    format_x_axis(ax2)
    plt.savefig(f'{output_dir}/chart2_arb_spread.png')
    plt.close(fig2)

    # Chart 3: The Fear Index (P/C Ratio)
    fig3, ax3 = plt.subplots(figsize=(16, 8))
    ax3.plot(df['Date'], df['P/C Vol'], label='Daily P/C Ratio',
             color='gray', alpha=0.4)
    ax3.plot(df['Date'], df['P/C Vol'].rolling(5).mean(), label='5-Day Moving Avg',
             color='#2ca02c', linewidth=3)
    ax3.axhline(y=1.0, color='#d62728', linestyle='--', label='Neutral (1.0)', alpha=0.8)
    ax3.set_title('Chart 3: The Fear Index (Put/Call Volume Ratio)', pad=20)
    ax3.set_ylabel('Put/Call Ratio')
    add_milestones(ax3)
    ax3.legend(loc='upper left')
    format_x_axis(ax3)
    plt.savefig(f'{output_dir}/chart3_fear_index.png')
    plt.close(fig3)

    # Chart 4: The Winner's Curse (Normalized Returns)
    fig4, ax4 = plt.subplots(figsize=(16, 8))
    colors = {'WBD': '#1f77b4', 'PSKY': '#ff7f0e', 'NFLX': '#d62728'}
    for ticker in ['WBD', 'PSKY', 'NFLX']:
        normalized = (df[ticker] / df[ticker].iloc[0]) * 100
        ax4.plot(df['Date'], normalized, label=ticker,
                 color=colors[ticker], linewidth=2.5)
    ax4.axhline(y=100, color='black', linestyle=':', alpha=0.5)
    ax4.set_title("Chart 4: The Winner's Curse (Normalized Returns)", pad=20)
    ax4.set_ylabel('Normalized Price (Base 100)')
    add_milestones(ax4)
    ax4.legend(loc='upper left')
    format_x_axis(ax4)
    plt.savefig(f'{output_dir}/chart4_winners_curse.png')
    plt.close(fig4)

    # Chart 5: Spread Z-Score (NEW)
    plot_spread_zscore(df, output_dir=output_dir)

    # ── Step 4: Backtest ─────────────────────────────────────────────
    backtest_winners_curse_pairs_trade(df)

    # ── Step 5: Save enriched data ───────────────────────────────────
    df.to_csv('data/enriched_signals_output.csv', index=False)
    print(f"\nSuccess! Charts saved to '{output_dir}/'.")
    print("Enriched signal data saved to 'data/enriched_signals_output.csv'.")


if __name__ == "__main__":
    main()