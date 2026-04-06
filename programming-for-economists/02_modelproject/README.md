# Labour Supply with a Kinked Budget Constraint

*BSc Economics, University of Copenhagen — Programming for Economists, Model Project (2025)*
*Group: Nawid Rasekh, Kasper Vinther, Mads Wittrup*

---

## The problem

Most income tax systems apply a higher marginal rate above an earnings threshold, creating a **kink** in the after-tax budget constraint. A standard interior-solution optimiser cannot handle this — the derivative of the budget set is undefined at the kink, so a naive minimiser may return the wrong answer or fail silently.

This project builds an analytically exact solver for the kinked constraint, then uses it to run population-level welfare simulations across 10,000 workers drawn from a log-normal productivity distribution.

---

## Model

**Preferences** — log utility in consumption with convex labour cost:

$$U(c, \ell) = \ln(c) - \frac{\nu \cdot \ell^{1+\varepsilon}}{1+\varepsilon}$$

**Tax schedule** — two-bracket piecewise-linear:

$$T(y) = \tau y + \zeta + \omega \cdot \max(y - \kappa, 0)$$

$\tau$ = flat rate, $\zeta$ = lump-sum, $\omega$ = top-bracket surcharge, $\kappa$ = income threshold.

**Social Welfare Function:**

$$\text{SWF} = \chi \cdot G^{\eta} + \sum_i U_i, \quad G = \sum_i T(y_i)$$

The SWF trades off individual utility against the value of public revenue — $\chi G^{\eta}$ captures the diminishing marginal value of government spending.

---

## The four-step FOC algorithm

For any worker with productivity $p$, the utility-maximising labour supply must lie in exactly one of three places. We check all three and take the maximum:

1. **Lower bracket interior** — solve $\varphi(p, \ell;\, \tau) = 0$ on $[0, \ell^{\kappa})$ via Brent's method
2. **At the kink** $\ell^{\kappa} = \kappa/(w \cdot p)$ — evaluate utility directly; the kink is a valid global maximum whenever the marginal utility of working shifts from positive to negative across the threshold
3. **Upper bracket interior** — solve $\varphi(p, \ell;\, \tau + \omega) = 0$ on $(\ell^{\kappa}, \ell_{\max}]$ via Brent's method

Exploiting the piecewise structure like this is **5–10× faster** than passing the full problem to `scipy.optimize.minimize_scalar`, because each Brent call searches a half-interval rather than the full domain.

---

## Key findings

**Bunching at the kink.** With $\omega = 0.2$, $\kappa = 9.0$, approximately **40% of the simulated workforce chooses exactly $\ell^{\kappa}$** — the computational analogue of the bunching masses documented by Saez (2010) for the US and Kleven & Waseem (2013) for Pakistan.

**Top tax and inequality:**

| Scenario | SWF | Tax revenue | Gini (consumption) |
|----------|-----|-------------|--------------------|
| Baseline ($\omega = 0$) | lower | lower | higher |
| Top tax ($\omega = 0.2$, $\kappa = 9.0$) | higher | higher | ~15% lower |

The top tax improves the SWF despite the labour-supply distortion because the public-good term $\chi G^{\eta}$ rises enough to outweigh the efficiency cost — redistribution wins on net.

**Optimal policy.** A grid search over $(\omega, \kappa) \in [0, 0.35] \times [6, 12]$ shows the calibrated parameters sit near but not exactly at the welfare maximum, leaving scope for marginal improvement.

---

## Files

| File | Contents |
|------|----------|
| `Worker.py` | Base class — parameters, income, quasi-linear utility, unconstrained FOC |
| `question3.py` | `TopTaxWorker` — log utility, two-bracket tax schedule, four-step FOC solver, SWF, Lorenz curve |
| `task3.py` | Analysis — population simulation, bunching plot, Lorenz/Gini analysis, welfare comparison, policy grid search, speed benchmark |
| `Modelproject_final.ipynb` | Main notebook — runs all analysis, displays results |

---

## How to run

```bash
pip install numpy matplotlib scipy seaborn
jupyter notebook Modelproject_final.ipynb
```

Run all cells top to bottom. No external data sources required — the population is generated synthetically with a fixed random seed (`RANDOM_SEED = 42`).
