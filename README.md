# Quant Project: WBD Deal Analysis & Event-Driven Trading

**Authors:** Vijay Hans, Kamil Sikora, Estella  

---

## Overview

This project analyzes market behavior during a merger battle involving Warner Bros. Discovery (WBD) and multiple potential acquirers.  

We focus on how implied volatility, arbitrage spreads, and options market sentiment reflect uncertainty around deal outcomes, and whether these signals can be exploited through event-driven strategies—specifically pairs trading under a “winner’s curse” framework.

---

## Key Observations

### Implied Volatility vs. Market Volatility
- WBD implied volatility consistently exceeded the VIX:
  - VIX: ~15–25  
  - WBD IV: ~30–80, with spikes above 100%
- Weak negative correlation:
  - **r = -0.239, p = 0.038**

**Interpretation:**  
Volatility was primarily driven by firm-specific event risk rather than macro conditions. After Netflix exited the deal, IV collapsed (**p ≈ 2e-35**), suggesting reduced uncertainty around upside outcomes.

---

### Persistent Arbitrage Spread
- Despite a $31 offer, WBD traded around $27–29 for extended periods.

**Interpretation:**  
Markets did not fully price in deal completion. The spread reflects perceived risks such as regulatory delays, deal failure, or changes in bid structure. It acts as a proxy for market confidence.

---

### Options Market Sentiment (Put/Call Ratio)
- Initially near 1, later rising sharply to ~5.65 (**p ≈ 0.03**)

**Interpretation:**  
Investor behavior shifted from upside speculation to downside protection. Increased demand for puts suggests rising concern about deal failure.

---

### Price Action Across Firms
- **WBD (target):** rises from ~$11 to ~$30 in stages  
- **Acquirers (NFLX, PSKY):** decline during negotiations  

**Interpretation:**  
Target firms capture takeover premiums, while acquirers face concerns about overpayment, leverage, and integration risk.

---

## Cross-Market Insights

- WBD valuation was driven primarily by takeover expectations rather than fundamentals  
- Markets priced a range of outcomes rather than a single expected deal price  
- Sentiment evolved from bid competition to downside hedging  

---

## Strategy: Winner’s Curse Pairs Trading

### Hypothesis
- Losing bidder rebounds after exiting the deal  
- Winning bidder underperforms due to financial and strategic burden  

### Trade Construction
- Long: losing bidder (NFLX)  
- Short: winning bidder (PSKY)  

---

## Methodology

### Cointegration
- Engle-Granger test:
  - **t = -4.2682, p = 0.0029**
- Indicates a mean-reverting relationship between the pair

### Hedge Ratio
- Static estimate:
  - β ≈ 0.18  
- Dynamic estimate:
  - 20-day rolling covariance/variance  

### Assumptions
- Risk-free rate: 4.5%  
- Short borrow cost: 0.5%  

---

## Results

### WBD Case (Short-Term)
- Sharpe ≈ 1.45 (non-annualized)  
- Max drawdown: ~5%  
- Slight outperformance relative to long NFLX  

### Dynamic Hedge Ratio
- Average β ≈ 0.8  
- Sharpe and PnL improved significantly  
- Stronger short-term performance, less consistent across cases  

---

## Testing on Other M&A Events

### Anadarko (2019)
- Strategy profitable early but dominated by shorting OXY  
- Alpha decays after ~1–3 months  

### Hulu (2023)
- Weak cointegration  
- Strategy underperforms  
- Long Comcast outperforms  

### Kansas City Southern (2021)
- Low hedge ratio  
- Strategy tracks stronger stock (CNI)  
- External shocks dominate price action  

---

## Entry and Exit Strategy

### Z-Score Thresholds
- Entry: 2σ deviation  
- Exit: 0.5σ reversal  

**Findings:**  
- High thresholds reduce trade frequency  
- Signals often lag momentum  
- Lower thresholds increase noise sensitivity  

---

## Limitations

- Small sample size of merger events  
- Limited time horizon (especially for WBD case)  
- Sensitivity to hedge ratio estimation  
- Entry/exit timing remains unresolved  

---

## Discussion: Why WBD Is Atypical

This deal deviates from standard M&A patterns due to:
- Aggressive and potentially hostile bidding behavior  
- Strategic resistance from WBD  
- Possible political and regulatory considerations  
- Non-economic motivations influencing participants  

These factors distort traditional signals such as volatility and spreads, making the case less clean than typical event-driven setups.

---

## Takeaways

- Event-driven alpha is short-lived  
- Fixed hedge ratios are more stable and reliable  
- Dynamic hedge ratios can improve returns but lack consistency  
- Pairs trading is most effective:
  - Over short horizons  
  - With strong cointegration  
  - In cleaner market structures  

---

## Future Work

- Expand dataset beyond U.S. merger events  
- Optimize entry/exit thresholds  
- Explore adaptive hedge ratios  
- Incorporate options-based strategies:
  - Short OTM calls (capped upside)  
  - Long OTM puts (downside hedge)  
- Model implied probabilities of deal completion  

---

## Implementation Notes

- Modular backtesting framework  
- Rolling statistical estimators  
- Measures taken to avoid look-ahead bias  

---

## Summary

This project examines how markets price uncertainty during merger events and evaluates whether that uncertainty can be systematically traded. The results suggest that while opportunities exist, they are short-lived, highly context-dependent, and sensitive to modeling choices.
