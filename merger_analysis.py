import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

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

def main():
    print("Loading and cleaning data...")
    options_df = pd.read_csv('data/wbd_options-overview-history-03-14-2026.csv', skipfooter=1, engine='python')
    stock_df = pd.read_csv('data/combined_WBD_PSKY_NFLX.csv')

    options_df['Date'] = pd.to_datetime(options_df['Date'])
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    options_df['Imp Vol'] = options_df['Imp Vol'].astype(str).str.replace('%', '').astype(float)
    
    df = pd.merge(options_df, stock_df, on='Date', how='inner')
    df = df.sort_values('Date')
    
    print("Generating fixed presentation charts...")
    setup_plot_style()
    output_dir = 'charts'
    os.makedirs(output_dir, exist_ok=True)

    # --- Chart 1: Idiosyncratic Risk (IV vs VIX) ---
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df['Date'], df['Imp Vol'], label='WBD Implied Volatility (%)', color='#9467bd', linewidth=2.5)
    ax1.plot(df['Date'], df['VIX'], label='Market Volatility (VIX)', color='#7f7f7f', linewidth=2.5)
    ax1.set_title('Chart 1: WBD Deal Uncertainty vs General Market Risk', pad=20)
    ax1.set_ylabel('Volatility (%)')
    ax1.legend(loc='upper right')
    format_x_axis(ax1)
    plt.savefig(f'{output_dir}/chart1_iv_vs_vix.png')
    plt.close(fig1)

    # --- Chart 2: The Arbitrage Spread ($31 vs WBD) ---
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(df['Date'], df['WBD'], label='WBD Stock Price', color='#1f77b4', linewidth=2.5)
    ax2.axhline(y=31.0, color='#d62728', linestyle='--', linewidth=2, label='Paramount Offer ($31.00)')
    ax2.set_title('Chart 2: The Arbitrage Spread & Market Doubt', pad=20)
    ax2.set_ylabel('Stock Price ($)')
    # Set y-limit slightly higher so the $31 line doesn't ride the top edge
    ax2.set_ylim(bottom=df['WBD'].min() * 0.9, top=35) 
    ax2.legend(loc='lower right')
    format_x_axis(ax2)
    plt.savefig(f'{output_dir}/chart2_arb_spread.png')
    plt.close(fig2)

    # --- Chart 3: The Fear Index (P/C Ratio) ---
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(df['Date'], df['P/C Vol'], label='Daily P/C Ratio', color='gray', alpha=0.4)
    # Add a 5-day moving average to smooth the daily noise
    ax3.plot(df['Date'], df['P/C Vol'].rolling(5).mean(), label='5-Day Moving Avg', color='#2ca02c', linewidth=3)
    ax3.axhline(y=1.0, color='#d62728', linestyle='--', label='Neutral (1.0)', alpha=0.8)
    ax3.set_title('Chart 3: The Fear Index (Put/Call Volume Ratio)', pad=20)
    ax3.set_ylabel('Put/Call Ratio')
    ax3.legend(loc='upper left')
    format_x_axis(ax3)
    plt.savefig(f'{output_dir}/chart3_fear_index.png')
    plt.close(fig3)

    # --- Chart 4: The Winner's Curse (Normalized Stocks) ---
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    colors = {'WBD': '#1f77b4', 'PSKY': '#ff7f0e', 'NFLX': '#d62728'}
    
    for ticker in ['WBD', 'PSKY', 'NFLX']:
        # Normalize to 100 at the start of the dataset
        normalized = (df[ticker] / df[ticker].iloc[0]) * 100
        ax4.plot(df['Date'], normalized, label=ticker, color=colors[ticker], linewidth=2.5)
        
    ax4.axhline(y=100, color='black', linestyle=':', alpha=0.5)
    ax4.set_title("Chart 4: The Winner's Curse (Normalized Returns)", pad=20)
    ax4.set_ylabel('Normalized Price (Base 100)')
    ax4.legend(loc='upper left')
    format_x_axis(ax4)
    plt.savefig(f'{output_dir}/chart4_winners_curse.png')
    plt.close(fig4)

    print(f"Success! 4 correctly formatted charts saved to '{output_dir}'.")

if __name__ == "__main__":
    main()