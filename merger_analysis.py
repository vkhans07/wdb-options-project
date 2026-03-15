import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import matplotlib.transforms as transforms

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
        '2025-12-5': 'WBD Announces Agreement w/Netflix',
        '2026-02-26': 'Netflix Exits Deal'
    }
    
    for date_str, label in milestones.items():
        date_obj = pd.to_datetime(date_str)
        ax.axvline(x=date_obj, color='black', linestyle='-.', alpha=0.6, linewidth=1.5)
        
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(date_obj, 0.95, f' {label}', transform=trans, 
                rotation=90, verticalalignment='top', horizontalalignment='right',
                fontsize=11, fontweight='bold', color='#333333')

def calculate_quant_metrics(df):
    """Calculates statistical proofs and engineers quantitative trading signals"""
    print("\n" + "="*50)
    print("STATISTICAL PROOFS FOR PRESENTATION SLIDES")
    print("="*50)
    
    # 1. Proof of Idiosyncratic Risk (Pearson Correlation)
    clean_vol = df[['Imp Vol', 'VIX']].dropna()
    corr, p_val_corr = stats.pearsonr(clean_vol['Imp Vol'], clean_vol['VIX'])
    print(f"[Slide 1] WBD IV vs VIX Correlation: {corr:.3f} (p-value: {p_val_corr:.4f})")
    
    # Define Regimes for T-Tests
    exit_date = pd.to_datetime('2026-02-26')
    before_exit = df[df['Date'] < exit_date]
    after_exit = df[df['Date'] >= exit_date]
    
    # 2. Proof of Volatility Crush (T-Test)
    t_stat_iv, p_val_iv = stats.ttest_ind(before_exit['Imp Vol'].dropna(), 
                                          after_exit['Imp Vol'].dropna(), 
                                          equal_var=False)
    print(f"[Slide 2] IV Crush: Before = {before_exit['Imp Vol'].mean():.1f}%, After = {after_exit['Imp Vol'].mean():.1f}% (p-value: {p_val_iv:.4e})")
    
    # 3. Sentiment Flip (Welch's T-Test on Put/Call Ratio)
    t_stat_pc, p_val_pc = stats.ttest_ind(before_exit['P/C Vol'].dropna(), 
                                          after_exit['P/C Vol'].dropna(), 
                                          equal_var=False)
    print(f"[Slide 3] P/C Ratio Flip: Before = {before_exit['P/C Vol'].mean():.2f}, After = {after_exit['P/C Vol'].mean():.2f} (p-value: {p_val_pc:.4e})")

    print("\n" + "="*50)
    print("ENGINEERING TRADING SIGNALS")
    print("="*50)
    
    # Signal 1: Volatility Risk Premium (VRP)
    df['WBD_Log_Ret'] = np.log(df['WBD'] / df['WBD'].shift(1))
    df['Realized_Vol'] = df['WBD_Log_Ret'].rolling(window=20).std() * np.sqrt(252) * 100 # Annualized 20-day RV
    df['VRP'] = df['Imp Vol'] - df['Realized_Vol']
    
    # Signal 2: Arbitrage Spread & Implied Probability
    df['Arb_Spread'] = 31.00 - df['WBD']
    unaffected_price = df['WBD'].iloc[0] # Baseline price before rumors
    
    # Implied Prob = (Current - Unaffected) / (Offer - Unaffected)
    implied_prob = (df['WBD'] - unaffected_price) / (31.00 - unaffected_price)
    df['Implied_Deal_Prob'] = np.clip(implied_prob, 0, 1) * 100 # Cap between 0% and 100%

    # Signal 3: Pairs Trade Hedge Ratio (Rolling Beta of NFLX vs PSKY)
    df['NFLX_Ret'] = df['NFLX'].pct_change()
    df['PSKY_Ret'] = df['PSKY'].pct_change()
    rolling_cov = df['NFLX_Ret'].rolling(window=20).cov(df['PSKY_Ret'])
    rolling_var = df['PSKY_Ret'].rolling(window=20).var()
    df['PSKY_NFLX_Beta'] = rolling_cov / rolling_var 
    
    print("\nLatest Signal Output (Last 3 Days of Data):")
    cols_to_show = ['Date', 'WBD', 'Arb_Spread', 'Implied_Deal_Prob', 'VRP', 'PSKY_NFLX_Beta']
    print(df[cols_to_show].tail(3).to_string(index=False))
    
    return df

def main():
    print("Loading and cleaning data...")
    options_df = pd.read_csv('data/wbd_options-overview-history-03-14-2026.csv', skipfooter=1, engine='python')
    stock_df = pd.read_csv('data/combined_WBD_PSKY_NFLX.csv')

    options_df['Date'] = pd.to_datetime(options_df['Date'])
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    options_df['Imp Vol'] = options_df['Imp Vol'].astype(str).str.replace('%', '').astype(float)
    
    df = pd.merge(options_df, stock_df, on='Date', how='inner')
    df = df.sort_values('Date')
    
    # ---> ADDED: Calculate all statistics and signals before charting
    df = calculate_quant_metrics(df)
    
    print("\nGenerating fixed presentation charts with milestones...")
    setup_plot_style()
    output_dir = 'charts'
    os.makedirs(output_dir, exist_ok=True)

    # --- Chart 1: Idiosyncratic Risk (IV vs VIX) ---
    fig1, ax1 = plt.subplots(figsize=(16,8))
    ax1.plot(df['Date'], df['Imp Vol'], label='WBD Implied Volatility (%)', color='#9467bd', linewidth=2.5)
    ax1.plot(df['Date'], df['VIX'], label='Market Volatility (VIX)', color='#7f7f7f', linewidth=2.5)
    ax1.set_title('Chart 1: WBD Deal Uncertainty vs General Market Risk', pad=20)
    ax1.set_ylabel('Volatility (%)')
    add_milestones(ax1)
    ax1.legend(loc='upper left')
    format_x_axis(ax1)
    plt.savefig(f'{output_dir}/chart1_iv_vs_vix.png')
    plt.close(fig1)

    # --- Chart 2: The Arbitrage Spread ($31 vs WBD) ---
    fig2, ax2 = plt.subplots(figsize=(16,8))
    ax2.plot(df['Date'], df['WBD'], label='WBD Stock Price', color='#1f77b4', linewidth=2.5)
    ax2.axhline(y=31.0, color='#d62728', linestyle='--', linewidth=2, label='Paramount Offer ($31.00)')
    ax2.fill_between(df['Date'], df['WBD'], 31.0, where=(df['WBD'] <= 31.0) & (df['Date'] >= pd.to_datetime('2026-02-26')), 
                     color='red', alpha=0.1, label='DOJ Risk Premium')
    ax2.set_title('Chart 2: The Arbitrage Spread & Market Doubt', pad=20)
    ax2.set_ylabel('Stock Price ($)')
    ax2.set_ylim(bottom=df['WBD'].min() * 0.9, top=35) 
    add_milestones(ax2)
    ax2.legend(loc='lower left')
    format_x_axis(ax2)
    plt.savefig(f'{output_dir}/chart2_arb_spread.png')
    plt.close(fig2)

    # --- Chart 3: The Fear Index (P/C Ratio) ---
    fig3, ax3 = plt.subplots(figsize=(16,8))
    ax3.plot(df['Date'], df['P/C Vol'], label='Daily P/C Ratio', color='gray', alpha=0.4)
    ax3.plot(df['Date'], df['P/C Vol'].rolling(5).mean(), label='5-Day Moving Avg', color='#2ca02c', linewidth=3)
    ax3.axhline(y=1.0, color='#d62728', linestyle='--', label='Neutral (1.0)', alpha=0.8)
    ax3.set_title('Chart 3: The Fear Index (Put/Call Volume Ratio)', pad=20)
    ax3.set_ylabel('Put/Call Ratio')
    add_milestones(ax3)
    ax3.legend(loc='upper left')
    format_x_axis(ax3)
    plt.savefig(f'{output_dir}/chart3_fear_index.png')
    plt.close(fig3)

    # --- Chart 4: The Winner's Curse (Normalized Stocks) ---
    fig4, ax4 = plt.subplots(figsize=(16,8))
    colors = {'WBD': '#1f77b4', 'PSKY': '#ff7f0e', 'NFLX': '#d62728'}
    
    for ticker in ['WBD', 'PSKY', 'NFLX']:
        normalized = (df[ticker] / df[ticker].iloc[0]) * 100
        ax4.plot(df['Date'], normalized, label=ticker, color=colors[ticker], linewidth=2.5)
        
    ax4.axhline(y=100, color='black', linestyle=':', alpha=0.5)
    ax4.set_title("Chart 4: The Winner's Curse (Normalized Returns)", pad=20)
    ax4.set_ylabel('Normalized Price (Base 100)')
    add_milestones(ax4)
    ax4.legend(loc='upper left')
    format_x_axis(ax4)
    plt.savefig(f'{output_dir}/chart4_winners_curse.png')
    plt.close(fig4)

    # Save the enriched data to CSV for final review
    df.to_csv('data/enriched_signals_output.csv', index=False)
    print(f"\nSuccess! Charts saved to '{output_dir}'. Enriched signal data saved to 'data/enriched_signals_output.csv'.")

if __name__ == "__main__":
    main()