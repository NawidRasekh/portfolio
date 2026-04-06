# Programming for Economists
### Computational Methods in Applied Macroeconomics and Microeconomics

*BSc Economics, University of Copenhagen — Programming for Economists (2025)*

Two projects combining formal economic theory with production-quality Python implementation: labour-supply modelling under nonlinear taxation, and computational general equilibrium.

---

## Projects at a glance

| # | Title | Methods | Key result |
|---|-------|---------|------------|
| 1 | [Labour Supply with a Kinked Budget Constraint](#1-labour-supply-with-a-kinked-budget-constraint) | Constrained optimisation, FOC root-finding, welfare analysis | ~40% of workers bunch at the kink; top tax reduces Gini by ~15%; FOC solver is 5–10× faster than black-box minimiser |
| 2 | [Applied General Equilibrium](#2-applied-general-equilibrium) | Walrasian equilibrium, tâtonnement, AS-AD macro model | Three equilibria under CES gross complements; Newton-Raphson converges in < 20 iterations vs. hundreds for tâtonnement |

---

## 1. Labour Supply with a Kinked Budget Constraint

**Folder:** [`02_modelproject/`](02_modelproject/)

### Problem

Most tax systems impose higher marginal rates on income above a threshold — creating a *kink* in the budget constraint. Standard interior-solution optimisation fails at the kink because the derivative of the budget set is discontinuous. This project implements an analytically exact solver and uses it to study the welfare effects of top taxation.

### Model

**Preferences:** Log utility in consumption with convex labour cost:
$$U(c, \ell) = \ln(c) - \frac{\nu \cdot \ell^{1+\varepsilon}}{1+\varepsilon}$$

**Tax schedule:** Two-bracket piecewise-linear:
$$T(y) = \tau y + \zeta + \omega \cdot \max(y - \kappa, 0)$$

where $\tau$ is the flat rate, $\zeta$ a lump-sum, $\omega$ the top-bracket surcharge, and $\kappa$ the income threshold. The kink in the net-of-tax budget constraint occurs at $\ell^{\kappa} = \kappa / (w \cdot p)$.

**Social Welfare Function:**
$$\text{SWF} = \chi \cdot G^{\eta} + \sum_i U_i, \quad G = \sum_i T(y_i)$$

### Four-step FOC algorithm

Rather than passing the problem to a black-box minimiser, we exploit the piecewise structure. For any productivity level $p$, the optimum must lie in exactly one of three places:

1. **Interior of the lower bracket** — solve the FOC $\varphi(p, \ell; \tau) = 0$ on $[0, \ell^{\kappa})$ via Brent's method
2. **At the kink** $\ell^{\kappa}$ — evaluate utility directly (the kink can be a local maximum even when interior solutions exist in both brackets)
3. **Interior of the upper bracket** — solve the FOC $\varphi(p, \ell; \tau + \omega) = 0$ on $(\ell^{\kappa}, \ell_{\max}]$

Then compare utilities across all three candidates and select the global maximum.

### Key findings

**Bunching:** With the calibrated parameters ($\omega = 0.2$, $\kappa = 9.0$), approximately **40% of workers along the productivity grid choose to work exactly $\ell^{\kappa}$ hours**, sacrificing potential earnings to avoid the top-bracket surcharge. This is the computational analogue of the empirical bunching masses documented by Saez (2010) for the US and Kleven & Waseem (2013) for Pakistan.

**Welfare and inequality:**

| Scenario | SWF | Total Tax Revenue | Gini (consumption) |
|----------|-----|------------------|--------------------|
| Baseline (no top tax, $\omega = 0$) | baseline | lower | higher |
| Top tax ($\omega = 0.2$, $\kappa = 9.0$) | +improved | higher | ~15% lower |

Introducing the top tax improves the SWF despite the labour-supply distortion, because the public-good term ($\chi G^{\eta}$) rises enough to outweigh the efficiency cost.

**Speed benchmark:** The FOC root-finding method is **5–10× faster** than `scipy.optimize.minimize_scalar` because it searches each half-interval separately, reducing the domain for each root-finder call.

**Optimal policy search:** A grid search over $(\omega, \kappa) \in [0, 0.35] \times [6, 12]$ confirms that the calibrated parameters are close to the welfare maximum, but not exactly at it — suggesting scope for marginal improvement through finer optimisation.

### Code architecture

```
02_modelproject/
├── Worker.py        Base class: parameters, budget arithmetic, quasi-linear utility
├── question3.py     TopTaxWorker: log utility, two-bracket tax, four-step FOC solver
└── task3.py         Analysis module: visualisations, bunching, welfare, policy search
```

---

## 2. Applied General Equilibrium

**Folder:** [`03_examproject/`](03_examproject/)

Three independent problems combining applied microeconomics, general equilibrium theory, and macroeconomic modelling.

---

### Problem 1: Danish House Prices

**Why real prices?** Nominal house price growth overstates real appreciation during high-inflation periods. Deflating by CPI isolates genuine changes in housing affordability and allows like-for-like comparison across regions and time.

**Key findings:**
- Real house prices (1992Q1 = 100) have grown substantially across all Danish provinces, but **regional divergence is pronounced** — Copenhagen-area municipalities have appreciated far faster than peripheral regions.
- The **correlation between initial price level and total growth is positive (r ≈ 0.6–0.7)**, indicating that already-expensive areas tended to appreciate most — consistent with a polarisation narrative.
- Several municipalities remain **below their pre-2008 peak** in real terms even after more than a decade, highlighting the long-lasting scarring from the financial crisis on peripheral housing markets.

---

### Problem 2: Walrasian Exchange Economy with CES Preferences

**Setup:** Two agents (A and B), two goods, CES utility with $\rho < 0$ (gross complements). Good 2 is the numeraire ($p_2 = 1$); equilibrium requires $e_1(p_1) = 0$.

**Algorithms implemented:**

| Method | Iterations to converge | Notes |
|--------|----------------------|-------|
| Tâtonnement $p^{k+1} = p^k + \nu \cdot e_1(p^k)$ | ~200–500 | Guaranteed convergence for gross substitutes; slower here |
| Dampened Newton-Raphson | < 20 | Uses numerical derivative; damping factor prevents overshoot |

**Multiple equilibria:** With gross complements ($\rho = -2$), the economy admits **three Walrasian equilibria**. The basin-of-attraction analysis (500 initial conditions) reveals that the two outer equilibria each attract roughly half the starting prices, while the middle equilibrium is unstable under tâtonnement — a classic result in general equilibrium theory.

**Edgeworth box:** All three equilibria are plotted in the Edgeworth box with corresponding indifference curves and budget lines, making it visually clear why each point clears the market.

---

### Problem 3: AS-AD Macroeconomic Model

**Model:** Standard discrete-time AS-AD with:
- **AD curve** derived from an IS-MP framework (Taylor rule + IS equation)
- **SRAS curve** with $\pi_t = \pi^e_t + \gamma(y_t - \bar{y})$
- **Adaptive expectations:** $\pi^e_t = \phi \cdot \pi_{t-1} + (1-\phi) \cdot \pi^e_{t-1}$

**Key results from simulation:**
- A one-time positive demand shock ($v_1 > 0$) raises both output and inflation on impact, with the SRAS curve shifting upward in subsequent periods as expectations adjust.
- **Higher shock persistence** ($\rho \to 1$) produces longer-lasting deviations from the $({\bar y}, \pi^*)$ long-run equilibrium.
- **Higher expectation rigidity** ($\phi \to 0$) slows convergence back to target, consistent with the central-bank concern that de-anchored inflation expectations are costly to reverse.

---

## Tools and methods

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Data wrangling | `pandas`, `numpy` |
| Statistical APIs | `dstapi` (Danmarks Statistik) |
| Optimisation | `scipy.optimize.minimize_scalar`, `scipy.optimize.root_scalar` (Brent's method) |
| Visualisation | `matplotlib`, `seaborn` |
| Economic methods | FOC root-finding on kinked constraints, tâtonnement, Newton-Raphson price adjustment, CES demand derivation, Gini/Lorenz inequality analysis, SWF maximisation |

---

## How to run

**Install dependencies:**

```bash
pip install pandas numpy matplotlib seaborn scipy dstapi openpyxl
```

**Run each project notebook:**

```bash
# Project 1 — Labour supply model
jupyter notebook 02_modelproject/Modelproject_final.ipynb

# Project 2 — General equilibrium exam
jupyter notebook 03_examproject/Examproject.ipynb
```

Note: The exam project fetches live house price and CPI data from Danmarks Statistik. An internet connection is required; results may differ slightly from the notebook outputs if the source data has been updated.

---

## Analytical approach and workflow

These projects use an **AI-augmented development workflow** — using Claude and GitHub Copilot for rapid prototyping, mathematical sanity-checking, and code review. The design philosophy throughout is:

1. **Theory first.** Each module starts from an analytical result (FOC, market-clearing condition, or statistical identity) and implements it directly. This avoids black-box computation where the code does not reflect economic reasoning.

2. **Separation of concerns.** Economic logic lives in `.py` modules (one class per model); notebooks are used for exploration and presentation only. This mirrors professional scientific computing practice and makes the models reusable.

3. **Benchmarking.** Where multiple solution methods are available (FOC vs. numerical, tâtonnement vs. Newton-Raphson), both are implemented and compared — because understanding *why* one method is faster or more reliable requires implementing the alternative.

4. **Reproducibility.** All data is fetched via public APIs with no manual preprocessing steps. Random seeds are set explicitly for population simulations.

---

*Nawid Rasekh — [GitHub](https://github.com/NawidRasekh) · University of Copenhagen, BSc Economics*
