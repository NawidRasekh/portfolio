"""
Worker base class
=================
Defines a representative worker who chooses how many hours to work given
a wage and a tax schedule.  The class is intentionally minimal: it stores
parameters, computes the budget constraint, and exposes utility.  The subclass
TopTaxWorker (question3.py) overrides these methods to add a realistic
piecewise top-tax bracket and implements the full FOC-based solver.

Preferences
-----------
The worker has quasi-linear utility:
    U(c, ℓ) = c − ν · ℓ^(1+ε) / (1+ε)

where c is consumption (= post-tax income under no savings), ℓ is hours
worked, ν > 0 scales the disutility of labour, and ε > 0 determines how
quickly marginal disutility rises with hours.

Quasi-linearity means there are no income effects on labour supply — only
substitution effects driven by the net wage.  This is a standard assumption
in public-finance models because it simplifies welfare analysis: lump-sum
taxes reduce utility one-for-one while proportional taxes distort the
labour-leisure margin.

Tax schedule
------------
Base schedule: T(y) = τ·y + ζ
    τ  — proportional (flat) tax rate on income y = w·p·ℓ
    ζ  — lump-sum tax (fixed regardless of income level)

Optional top bracket: when κ is defined, income above κ is also taxed at ω:
    T(y) = τ·y + ζ + ω·max(y − κ, 0)

This piecewise-linear schedule creates a kink in the budget constraint at
y = κ, which is the central feature analysed in question3.py / task3.py.
"""

from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize_scalar, root_scalar


class WorkerClass:
    """
    Representative worker with a (potentially) kinked tax schedule.

    Parameters live in self.par (SimpleNamespace) so they can be adjusted
    externally without subclassing, following the course convention.
    """

    def __init__(self, par=None):
        # Always initialise defaults first, then override with caller's values
        self.setup_worker()
        if par is not None:
            for k, v in par.items():
                self.par.__dict__[k] = v

    def setup_worker(self):
        """Set default calibration values."""
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # Preferences
        par.nu      = 0.015  # disutility weight on labour hours
        par.epsilon = 1.0    # curvature of disutility (ε=1 → quadratic cost)

        # Labour market
        par.w       = 1.0                        # wage rate (normalised to 1)
        par.ps      = np.linspace(0.5, 3.0, 100) # productivity grid for analysis
        par.ell_max = 16.0                        # maximum daily hours

        # Tax parameters
        par.tau   = 0.50   # flat proportional tax rate
        par.zeta  = 0.10   # lump-sum (non-distortionary) tax
        par.kappa = np.nan # income threshold for top bracket; NaN = no top tax
        par.omega = 0.20   # top-bracket surcharge rate (applied above κ)

    # ------------------------------------------------------------------
    # Budget constraint
    # ------------------------------------------------------------------

    def income(self, p, ell):
        """Pre-tax labour income: y = w · p · ℓ."""
        return self.par.w * p * ell

    def tax(self, pre_tax_income):
        """
        Total tax paid on pre-tax income y.

        Applies the flat rate τ and lump sum ζ.  If the top-bracket threshold
        κ is defined (not NaN), also charges ω on income exceeding κ.
        """
        par = self.par
        tax = par.tau * pre_tax_income + par.zeta

        # Top-bracket surcharge — only active when κ is set
        if not np.isnan(par.kappa):
            tax += par.omega * np.fmax(pre_tax_income - par.kappa, 0)

        return tax

    def post_tax_income(self, p, ell):
        """Post-tax consumption: c = y − T(y).  (No savings in this model.)"""
        pre_tax = self.income(p, ell)
        return pre_tax - self.tax(pre_tax)

    def max_post_tax_income(self, p):
        """Maximum attainable consumption when working ℓ_max hours."""
        return self.post_tax_income(p, self.par.ell_max)

    def get_min_ell(self, p):
        """
        Minimum hours needed to cover the lump-sum tax ζ.

        Below this level consumption would be negative, so the worker would
        never rationally choose ℓ < get_min_ell(p).
        """
        par = self.par
        min_ell = par.zeta / (par.w * p * (1 - par.tau))
        return np.fmax(min_ell, 0.0) + 1e-8

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def utility(self, c, ell):
        """
        Quasi-linear utility: U = c − ν · ℓ^(1+ε) / (1+ε).

        At the interior optimum the first-order condition sets the real net
        wage equal to the marginal disutility of labour:
            (1−τ)·w·p / c = ν · ℓ^ε      [standard bracket]
            (1−τ−ω)·w·p / c = ν · ℓ^ε   [top bracket]
        TopTaxWorker.optimal_choice_foc_kink implements this logic.
        """
        par = self.par
        return c - par.nu * (ell ** (1 + par.epsilon)) / (1 + par.epsilon)

    def value_of_choice(self, p, ell):
        """Utility attained by choosing ℓ given productivity p."""
        c = self.post_tax_income(p, ell)
        return self.utility(c, ell)

    # ------------------------------------------------------------------
    # Optimisation stubs (implemented in TopTaxWorker)
    # ------------------------------------------------------------------

    def optimal_choice(self, p):
        """Numerical optimisation over ℓ. Implemented in subclass."""
        par = self.par
        opt = SimpleNamespace()
        pass

    def FOC(self, p, ell):
        """First-order condition residual. Implemented in subclass."""
        par = self.par
        pass

    def optimal_choice_FOC(self, p):
        """FOC root-finding optimisation. Implemented in subclass."""
        par = self.par
        opt = SimpleNamespace()
        pass
        return opt
