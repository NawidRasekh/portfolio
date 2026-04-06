# Applied General Equilibrium

*BSc Economics, University of Copenhagen — Programming for Economists, Exam Project (2025)*
*Group: Nawid Rasekh, Kasper Vinther, Mads Wittrup*

Three problems in applied microeconomics, general equilibrium theory, and macroeconomic modelling — each implemented in a self-contained Python module.

---

## Problem 1 — Danish House Prices

**Contributed by: Nawid Rasekh**

**File:** `problem1_analysis.py` · **Data:** `BM010_houses.xlsx`, Danmarks Statistik APIs `EJ56` and `PRIS113`

### Why real prices?

Nominal house price growth conflates genuine appreciation with general inflation. Deflating by CPI (re-based to 1992Q1 = 100) isolates true changes in housing affordability and makes cross-region, cross-decade comparisons valid.

### Key findings

- Real prices have grown across all Danish provinces since 1992, but **regional divergence is stark** — Copenhagen-area municipalities have appreciated far faster than peripheral regions.
- The **correlation between initial price level (1992) and total real growth is r ≈ 0.6–0.7** — already-expensive areas appreciated most, consistent with a housing-market polarisation narrative.
- Several peripheral municipalities are still **trading below their pre-2008 peak in real terms**, more than a decade after the financial crisis — the correction was not symmetrically reversed.

### Methods

CPI deflation via `PRIS113` (monthly, resampled to quarterly), re-based to match the house price index at 1992Q1. Municipality-level data from `BM010_houses.xlsx` used for the scatter analysis and rolling-average crisis-recovery chart.

---

## Problem 2 — Walrasian Exchange Economy with CES Preferences

**Contributed by: Kasper Vinther**

**File:** `ExchangeEconomyModel.py`

### Setup

Two agents (A and B), two goods. CES utility with $\rho < 0$ (gross complements):

$$u_i(x_1, x_2) = \left(\alpha_i x_1^{\rho} + \beta_i x_2^{\rho}\right)^{1/\rho}$$

Good 2 is the numeraire ($p_2 = 1$); Walrasian equilibrium requires excess demand $e_1(p_1) = 0$.

### Algorithms

| Method | Convergence | Notes |
|--------|-------------|-------|
| Tâtonnement: $p^{k+1} = p^k + \nu \cdot e_1(p^k)$ | ~200–500 iterations | Reliable for gross substitutes; slower under complements |
| Dampened Newton-Raphson | < 20 iterations | Numerical derivative + damping factor prevents overshoot |

### Key finding

With $\rho = -2$ (gross complements), the economy admits **three Walrasian equilibria**. A basin-of-attraction analysis (500 initial price conditions) shows the two outer equilibria each attract roughly half the starting values; the middle equilibrium is unstable under tâtonnement — a textbook result in GE theory made concrete.

---

## Problem 3 — AS-AD Macroeconomic Model

**Contributed by: Mads Wittrup**

**File:** `ASADModel.py`

### Model

Discrete-time AS-AD in the $(y_t, \pi_t)$ space:

- **AD curve** (from IS-MP / Taylor rule): $\pi_t = \pi^* - \frac{1}{\alpha}\left[(y_t - \bar{y}) - z_t\right]$
- **SRAS curve**: $\pi_t = \pi^e_t + \gamma(y_t - \bar{y})$
- **Adaptive expectations**: $\pi^e_t = \phi \cdot \pi_{t-1} + (1 - \phi) \cdot \pi^e_{t-1}$

Equilibrium each period is solved analytically by substituting SRAS into AD, treating $\pi^e_t$ and $v_t$ as given.

### Key results

- A one-time positive demand shock raises output and inflation on impact; the SRAS shifts upward in subsequent periods as expectations adjust upward.
- **Higher shock persistence** ($\rho \to 1$) prolongs deviations from $(\bar{y}, \pi^*)$.
- **Higher expectation rigidity** ($\phi \to 0$) slows the return to target — the mechanism behind the central-bank concern that de-anchored expectations are costly to reverse.

---

## How to run

```bash
pip install numpy pandas matplotlib scipy dstapi openpyxl
jupyter notebook Examproject.ipynb
```

Run all cells top to bottom. Problem 1 fetches live data from Danmarks Statistik — an internet connection is required, and results may differ slightly if the source series has been updated.
