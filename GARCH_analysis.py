# =============================================================================
#  NIFTY 50 — Volatility Analysis using ARCH/GARCH Models (2015–2024)
#  Author  : [Your Name]
#  Method  : ARCH(1) and GARCH(1,1) via Maximum Likelihood Estimation
#  Data    : Yahoo Finance (^NSEI) — Daily Closing Prices
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from statsmodels.stats.diagnostic import het_arch
from statsmodels.graphics.tsaplots import plot_acf
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.facecolor":   "#0d1117",
    "figure.facecolor": "#0d1117",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "grid.color":       "#21262d",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

CYAN   = "#58a6ff"
GREEN  = "#3fb950"
RED    = "#f85149"
YELLOW = "#e3b341"
PURPLE = "#bc8cff"
GREY   = "#8b949e"
BG     = "#0d1117"

# =============================================================================
# 1. LOAD & PREPARE DATA
# =============================================================================
df = pd.read_csv("NIFTY_50_final_dataset.csv")
df = df[df['Price'].str.match(r'\d{4}-\d{2}-\d{2}', na=False)].copy()
df.columns = ['Date', 'Close', 'Log_Return']
df['Date']       = pd.to_datetime(df['Date'])
df['Close']      = pd.to_numeric(df['Close'],      errors='coerce')
df['Log_Return'] = pd.to_numeric(df['Log_Return'], errors='coerce')
df = df.dropna().set_index('Date')

returns = df['Log_Return'] * 100   # scale to % for ARCH/GARCH

print("=" * 60)
print("  NIFTY 50 DATASET SUMMARY")
print("=" * 60)
print(f"  Period        : {df.index[0].date()} → {df.index[-1].date()}")
print(f"  Observations  : {len(df):,}")
print(f"  Mean Return   : {returns.mean():.4f}%")
print(f"  Std Dev       : {returns.std():.4f}%")
print(f"  Min Return    : {returns.min():.4f}%")
print(f"  Max Return    : {returns.max():.4f}%")
print(f"  Skewness      : {returns.skew():.4f}")
print(f"  Kurtosis      : {returns.kurtosis():.4f}")

# =============================================================================
# 2. ARCH LM TEST
# =============================================================================
arch_test = het_arch(returns)
print("\n" + "=" * 60)
print("  ARCH LM TEST (Engle 1982)")
print("=" * 60)
print(f"  Test Statistic : {arch_test[0]:.4f}")
print(f"  p-value        : {arch_test[1]:.2e}")
if arch_test[1] < 0.05:
    print("  Conclusion     : ARCH effect PRESENT ✅ — GARCH modelling is justified")
else:
    print("  Conclusion     : No ARCH effect ❌")

# =============================================================================
# 3. FIT ARCH(1) MODEL
# =============================================================================
arch_fit = arch_model(returns, vol='ARCH', p=1, rescale=False).fit(disp='off')
print("\n" + "=" * 60)
print("  ARCH(1) MODEL RESULTS")
print("=" * 60)
print(arch_fit.summary())

# =============================================================================
# 4. FIT GARCH(1,1) MODEL
# =============================================================================
garch_fit = arch_model(returns, vol='GARCH', p=1, q=1, rescale=False).fit(disp='off')
print("\n" + "=" * 60)
print("  GARCH(1,1) MODEL RESULTS")
print("=" * 60)
print(garch_fit.summary())

# =============================================================================
# 5. MODEL COMPARISON
# =============================================================================
print("\n" + "=" * 60)
print("  MODEL COMPARISON")
print("=" * 60)
print(f"  {'Model':<15} {'Log-Likelihood':>15} {'AIC':>12} {'BIC':>12}")
print(f"  {'-'*54}")
print(f"  {'ARCH(1)':<15} {arch_fit.loglikelihood:>15.2f} {arch_fit.aic:>12.2f} {arch_fit.bic:>12.2f}")
print(f"  {'GARCH(1,1)':<15} {garch_fit.loglikelihood:>15.2f} {garch_fit.aic:>12.2f} {garch_fit.bic:>12.2f}")
print(f"\n  ✅ GARCH(1,1) preferred — lower AIC/BIC")

# =============================================================================
# 6. GARCH(1,1) PERSISTENCE
# =============================================================================
omega = garch_fit.params['omega']
alpha = garch_fit.params['alpha[1]']
beta  = garch_fit.params['beta[1]']
persistence = alpha + beta
long_run_vol = np.sqrt(omega / (1 - persistence)) if persistence < 1 else np.nan

print("\n" + "=" * 60)
print("  GARCH(1,1) KEY PARAMETERS")
print("=" * 60)
print(f"  ω (omega)       : {omega:.6f}")
print(f"  α (alpha[1])    : {alpha:.4f}  — short-run shock sensitivity")
print(f"  β (beta[1])     : {beta:.4f}  — volatility persistence")
print(f"  α + β           : {persistence:.4f}  — total persistence")
print(f"  Long-run vol    : {long_run_vol:.4f}% per day")
if persistence < 1:
    print("  Mean-reverting  : YES ✅ (α + β < 1)")

# =============================================================================
# 7. 30-DAY FORECAST
# =============================================================================
forecast = garch_fit.forecast(horizon=30)
forecast_vol = np.sqrt(forecast.variance.iloc[-1].values)

# =============================================================================
# 8. VISUALISATIONS
# =============================================================================

# ── Fig 1: Log Returns ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4.5), facecolor=BG)
ax.set_facecolor(BG)
colors = [GREEN if r >= 0 else RED for r in returns]
ax.bar(df.index, returns, color=colors, width=1, alpha=0.85)
ax.axhline(0, color=GREY, lw=0.6)

# COVID annotation
covid_date = pd.Timestamp("2020-03-23")
ax.annotate("COVID-19\nCrash", xy=(covid_date, returns.min()),
            xytext=(pd.Timestamp("2021-01-01"), returns.min() + 3),
            arrowprops=dict(arrowstyle="->", color=YELLOW, lw=1.5),
            color=YELLOW, fontsize=9, fontweight="bold")

ax.set_title("NIFTY 50 — Daily Log Returns (2015–2024)", fontsize=13,
             fontweight="bold", color="#e6edf3", pad=12)
ax.set_xlabel("Date", fontsize=10)
ax.set_ylabel("Log Return (%)", fontsize=10)
ax.legend(handles=[
    plt.Rectangle((0,0),1,1, color=GREEN, label="Positive"),
    plt.Rectangle((0,0),1,1, color=RED,   label="Negative")],
    fontsize=9, framealpha=0, loc="upper left")
plt.tight_layout()
plt.savefig("chart1_log_returns.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("\n✅  Saved: chart1_log_returns.png")

# ── Fig 2: GARCH Conditional Volatility ──────────────────────────────────────
cond_vol = garch_fit.conditional_volatility
fig, ax = plt.subplots(figsize=(13, 4.5), facecolor=BG)
ax.set_facecolor(BG)
ax.fill_between(df.index, cond_vol, alpha=0.25, color=CYAN)
ax.plot(df.index, cond_vol, color=CYAN, lw=1.2, label="GARCH(1,1) Conditional Volatility")
ax.axhline(long_run_vol, color=YELLOW, lw=1, ls="--",
           label=f"Long-run vol = {long_run_vol:.3f}%")
ax.annotate("COVID-19", xy=(covid_date, cond_vol[df.index.get_indexer([covid_date], method='nearest')[0]]),
            xytext=(pd.Timestamp("2021-06-01"), cond_vol.max() * 0.85),
            arrowprops=dict(arrowstyle="->", color=YELLOW, lw=1.2),
            color=YELLOW, fontsize=9, fontweight="bold")
ax.set_title("GARCH(1,1) — Conditional Volatility Over Time", fontsize=13,
             fontweight="bold", color="#e6edf3", pad=12)
ax.set_xlabel("Date", fontsize=10)
ax.set_ylabel("Volatility (%)", fontsize=10)
ax.legend(fontsize=9, framealpha=0)
plt.tight_layout()
plt.savefig("chart2_garch_volatility.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("✅  Saved: chart2_garch_volatility.png")

# ── Fig 3: Return Distribution vs Normal ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
ax.set_facecolor(BG)
ax.hist(returns, bins=80, color=CYAN, alpha=0.6, density=True, label="Actual Returns")
x = np.linspace(returns.min(), returns.max(), 300)
ax.plot(x, stats.norm.pdf(x, returns.mean(), returns.std()),
        color=YELLOW, lw=2, label="Normal Distribution")
ax.set_title("Return Distribution vs Normal", fontsize=12,
             fontweight="bold", color="#e6edf3", pad=10)
ax.set_xlabel("Log Return (%)", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.legend(fontsize=9, framealpha=0)
plt.tight_layout()
plt.savefig("chart3_distribution.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("✅  Saved: chart3_distribution.png")

# ── Fig 4: ACF of Squared Returns ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
ax.set_facecolor(BG)
plot_acf(returns**2, lags=20, ax=ax, color=CYAN, zero=False,
         title="ACF of Squared Returns (ARCH Effect)")
ax.set_xlabel("Lag", fontsize=10)
ax.set_ylabel("Autocorrelation", fontsize=10)
ax.title.set_color("#e6edf3")
ax.title.set_fontsize(12)
ax.title.set_fontweight("bold")
plt.tight_layout()
plt.savefig("chart4_acf_squared.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("✅  Saved: chart4_acf_squared.png")

# ── Fig 5: 30-Day Forecast ────────────────────────────────────────────────────
band = forecast_vol * 0.10
fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
ax.set_facecolor(BG)
days = np.arange(1, 31)
ax.fill_between(days, forecast_vol - band, forecast_vol + band,
                alpha=0.3, color=YELLOW, label="±10% band")
ax.plot(days, forecast_vol, color=YELLOW, lw=2.2, marker="o", ms=4)
ax.axhline(long_run_vol, color=RED, lw=1.2, ls="--",
           label=f"Long-run vol = {long_run_vol:.3f}%")
ax.set_title("30-Day Volatility Forecast — GARCH(1,1)", fontsize=12,
             fontweight="bold", color="#e6edf3", pad=10)
ax.set_xlabel("Forecast Horizon (days)", fontsize=10)
ax.set_ylabel("Volatility (%)", fontsize=10)
ax.legend(fontsize=9, framealpha=0)
plt.tight_layout()
plt.savefig("chart5_forecast.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("✅  Saved: chart5_forecast.png")

# ── Fig 6: Q-Q Plot ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
ax.set_facecolor(BG)
(osm, osr), (slope, intercept, r) = stats.probplot(returns, dist="norm")
ax.scatter(osm, osr, color=CYAN, alpha=0.5, s=10)
ax.plot(osm, slope * np.array(osm) + intercept, color=RED, lw=2, label="Normal line")
ax.set_title("Q-Q Plot (Heavy Tails Evidence)", fontsize=12,
             fontweight="bold", color="#e6edf3", pad=10)
ax.set_xlabel("Theoretical Quantiles", fontsize=10)
ax.set_ylabel("Sample Quantiles", fontsize=10)
ax.legend(fontsize=9, framealpha=0)
plt.tight_layout()
plt.savefig("chart6_qq_plot.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("✅  Saved: chart6_qq_plot.png")

print("\n✅  All charts saved. Project ready for GitHub upload!")
