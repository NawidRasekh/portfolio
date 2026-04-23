# Programming for Economists — Group Contributions

*BSc Economics, University of Copenhagen — Programming for Economists, Exam Project (2025)*
*Group: Nawid Rasekh, Kasper Vinther, Mads Wittrup*

---

This folder holds the two exam-project problems where the primary
implementation was done by my groupmates. They are kept in the portfolio
because I contributed to problem design, report writing, and joint review
across the full exam, and because the economic content complements the
other projects in the portfolio.

The problems I led — **Danish house prices** (deflation and regional
divergence) — and the full **labour-supply model** that we built together
— live one level up, in the portfolio root, with their own READMEs.

## Problems in this folder

### [exchange-economy-ces/](./exchange-economy-ces/)
**Walrasian Exchange Economy with CES Preferences**
*Primary author: Kasper Vinther*

Two agents, two goods, CES utility with $\rho < 0$ (gross complements).
Implements tâtonnement and dampened Newton-Raphson price adjustment,
locates three Walrasian equilibria, and visualises the basins of attraction
and Edgeworth box.

`Python` · `NumPy` · `SciPy` · `matplotlib` · `general equilibrium`

### [as-ad-macro-model/](./as-ad-macro-model/)
**AS-AD Macroeconomic Model with Adaptive Expectations**
*Primary author: Mads Wittrup*

Discrete-time AS-AD model derived from IS-MP. Simulates responses to demand
shocks with persistence $\rho$ and adaptive expectations parameter $\phi$,
and shows how de-anchored inflation expectations slow convergence back to
the policy target.

`Python` · `NumPy` · `matplotlib` · `macroeconomic simulation`

---

## Course context

These problems are from the **Exam Project** in *Programming for Economists*
(UCPH, spring 2025), submitted jointly by all three group members.
The group-level report and all code reviews happened collectively;
the "primary author" label marks who wrote the first working version
of each problem's code. The notebooks in each subfolder are clean extracts
from the combined exam submission, so outputs are embedded and reproducible.
