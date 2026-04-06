"""
Question 3: Worker with a Top-Tax Bracket
==========================================
Extends the WorkerClass with a piecewise-linear tax schedule that adds a
surcharge on income above a threshold κ.  The resulting kink in the budget
constraint makes standard interior-solution optimisation unreliable (the
FOC may not hold on the boundary), so we implement a four-step algorithm
that explicitly checks all three candidate optima:

    1. Interior solution in the LOWER bracket  (y < κ)
    2. Corner solution AT the kink              (y = κ)
    3. Interior solution in the UPPER bracket  (y > κ)

The algorithm then picks whichever gives the highest utility.

Economic motivation
-------------------
In the Danish tax system (and most Scandinavian countries), marginal tax
rates increase sharply once income crosses certain thresholds.  This creates
a strong incentive for some workers to "bunch" at the kink — i.e. to work
exactly enough hours so that their income equals κ — avoiding the top-rate
surcharge.  Empirical bunching at tax kinks is well-documented (Saez 2010)
and this model replicates the mechanism analytically.

Utility function (TopTaxWorker)
--------------------------------
U(c, ℓ) = ln(c) − ν · ℓ^(1+ε) / (1+ε)

Note: the base WorkerClass uses a *linear* utility in c.  Here we switch to
log utility in c, making the model more standard in the labour-supply
literature and ensuring a genuine interior optimum exists in each bracket.
The log specification implies diminishing marginal utility of consumption,
which generates realistic labour-supply responses.

Tax schedule
------------
T(y) = τ·y + ζ + ω·max(y − κ, 0)

Effective marginal rates:
    Below κ:  τ
    Above κ:  τ + ω   (the "top rate")

The kink threshold in labour-supply space is:
    ℓ_kink = κ / (w·p)    [hours at which income exactly equals κ]
"""

import numpy as np
from scipy.optimize import minimize_scalar, root_scalar
from Worker import WorkerClass


class TopTaxWorker(WorkerClass):
    """
    Worker subject to a two-bracket piecewise-linear income tax.

    Inherits all parameter setup and budget arithmetic from WorkerClass;
    overrides utility (log instead of linear) and tax to add the top bracket,
    then implements two solvers:
        optimal_choice_numerical  — black-box scipy minimiser
        optimal_choice_foc_kink   — analytical four-step FOC algorithm
    """

    def __init__(self, par=None):
        super().__init__(par)

        # Ensure top-tax parameters are initialised to sensible defaults
        # if the parent class left them as NaN/None
        if self.par.kappa is None or np.isnan(self.par.kappa):
            self.par.kappa = 9.0   # income threshold in model units
        if self.par.omega is None:
            self.par.omega = 0.2   # 20% surcharge on income above κ

    # ------------------------------------------------------------------
    # Utility: log specification
    # ------------------------------------------------------------------

    def utility(self, c, ell):
        """
        Log utility in consumption: U = ln(c) − ν · ℓ^(1+ε) / (1+ε).

        Log utility ensures the marginal utility of consumption is 1/c,
        which gives the clean FOC:   (1−marginal_tax)·w·p / c = ν · ℓ^ε
        This is the standard Frisch labour-supply equation in the literature.
        """
        par = self.par
        if np.isscalar(c):
            if c <= 0:
                return -np.inf
        else:
            c = np.maximum(c, 1e-8)  # guard against log(0) in vectorised calls

        return (np.log(c)
                - par.nu * (ell ** (1.0 + par.epsilon)) / (1.0 + par.epsilon))

    # ------------------------------------------------------------------
    # Tax: adds top bracket to base schedule
    # ------------------------------------------------------------------

    def tax(self, pre_tax_income):
        """
        Two-bracket tax: T(y) = τ·y + ζ + ω·max(y − κ, 0).

        The top-bracket term ω·max(y−κ,0) equals zero when y ≤ κ and rises
        linearly above κ, creating the kink in the net-of-tax budget line.
        """
        par = self.par
        standard_tax = par.tau * pre_tax_income + par.zeta
        top_tax      = par.omega * np.maximum(pre_tax_income - par.kappa, 0)
        return standard_tax + top_tax

    def income(self, p, ell):
        """Pre-tax labour income: y = w · p · ℓ."""
        return self.par.w * p * ell

    def post_tax_income(self, p, ell):
        """Consumption = pre-tax income minus total tax."""
        pre_tax = self.income(p, ell)
        return pre_tax - self.tax(pre_tax)

    # ------------------------------------------------------------------
    # Solver 1: numerical (benchmark for speed comparison)
    # ------------------------------------------------------------------

    def optimal_choice_numerical(self, p):
        """
        Find optimal ℓ by passing the negative utility to scipy's
        minimize_scalar with bounded search over [0, ℓ_max].

        This is the "black box" approach — correct but slow because the
        optimiser does not exploit the piecewise structure.  Used as a
        benchmark to demonstrate the speed advantage of the FOC method.

        Returns
        -------
        (ell_star, c_star, u_star)
        """
        par = self.par

        def obj(ell):
            c = self.post_tax_income(p, ell)
            if c <= 0:
                return np.inf   # infeasible — strongly discourage
            return -self.utility(c, ell)

        res      = minimize_scalar(obj, bounds=(0, par.ell_max), method='bounded')
        ell_star = res.x
        c_star   = self.post_tax_income(p, ell_star)
        u_star   = -res.fun

        return ell_star, c_star, u_star

    # ------------------------------------------------------------------
    # FOC helper: signed residual for plotting and root-finding
    # ------------------------------------------------------------------

    def foc_error(self, p, ell, type='standard'):
        """
        Compute the FOC residual (MRS − net real wage) at (p, ℓ).

        The FOC for an interior solution is:
            (1 − effective_tax) · w · p / c  =  ν · ℓ^ε
        Rearranging:  φ = (1−τ)·w·p / c − ν·ℓ^ε = 0

        We return φ so that:
            φ > 0  ⟹  net wage > MRS  ⟹  worker wants to work MORE
            φ < 0  ⟹  net wage < MRS  ⟹  worker wants to work LESS
            φ = 0  ⟹  interior optimum

        Parameters
        ----------
        type : 'standard' (lower bracket, rate τ) or 'top' (upper bracket, rate τ+ω)
            Determines which effective tax rate is used to compute the FOC,
            allowing us to extend each FOC curve across the full ℓ range for
            plotting purposes (even into regions where that bracket doesn't apply).
        """
        par    = self.par
        income = par.w * p * ell

        if type == 'standard':
            tax              = par.tau * income + par.zeta
            effective_rate   = par.tau
        elif type == 'top':
            tax              = (par.tau * income + par.zeta
                                + par.omega * (income - par.kappa))
            effective_rate   = par.tau + par.omega

        c = income - tax
        if c <= 0:
            return np.nan

        mrs      = par.nu * (ell ** par.epsilon)
        net_wage = (1 - effective_rate) * par.w * p / c

        return net_wage - mrs   # zero at the interior optimum

    # ------------------------------------------------------------------
    # Solver 2: analytical four-step FOC algorithm (main solver)
    # ------------------------------------------------------------------

    def optimal_choice_foc_kink(self, p):
        """
        Analytically find the utility-maximising ℓ for a worker with
        productivity p, using the four-step algorithm for kinked budgets.

        Algorithm
        ---------
        Step 1 – Lower bracket interior solution
            Solve FOC for τ on the interval [0, ℓ_kink) using Brent's method.
            If root exists and is in range, record the resulting utility.

        Step 2 – Kink solution
            Evaluate utility at ℓ = ℓ_kink (boundary between brackets).
            Workers may prefer this even if an interior solution exists in
            one bracket, because the kink is where the budget constraint
            changes slope and can be a local maximum of utility.

        Step 3 – Upper bracket interior solution
            Solve FOC for (τ + ω) on the interval (ℓ_kink, ℓ_max] using
            Brent's method.

        Step 4 – Pick the global maximum
            Compare utilities from steps 1-3 and return the best option
            along with a label indicating which regime it belongs to.

        Returns
        -------
        (ell_star, u_star, regime_label)
            regime_label ∈ {'lower_bracket', 'kink', 'upper_bracket'}
        """
        par = self.par

        # Labour hours at which income exactly equals the top-tax threshold
        ell_kink = par.kappa / (par.w * p)

        # ---- Step 1: lower bracket interior solution ----
        def foc_lower(ell):
            # Interior FOC for y < κ: net wage = MRS
            income = par.w * p * ell
            tax    = par.tau * income + par.zeta
            c      = income - tax
            if c <= 0:
                return -np.inf
            mrs          = par.nu * (ell ** par.epsilon) * c
            real_wage_net = (1 - par.tau) * par.w * p
            return real_wage_net - mrs

        ell_b, u_b = np.nan, -np.inf
        if ell_kink > 0:
            try:
                res_b = root_scalar(foc_lower, bracket=[1e-6, ell_kink],
                                    method='brentq')
                if res_b.converged:
                    ell_b = res_b.root
                    u_b   = self.utility(self.post_tax_income(p, ell_b), ell_b)
            except ValueError:
                pass   # no root in bracket — optimum is elsewhere

        # ---- Step 2: kink solution ----
        ell_k = ell_kink
        u_k   = (self.utility(self.post_tax_income(p, ell_k), ell_k)
                 if ell_k <= par.ell_max else -np.inf)

        # ---- Step 3: upper bracket interior solution ----
        def foc_upper(ell):
            # Interior FOC for y > κ: net wage at rate (τ+ω) = MRS
            c = self.post_tax_income(p, ell)
            if c <= 0:
                return -np.inf
            mrs           = par.nu * (ell ** par.epsilon) * c
            real_wage_net = (1 - par.tau - par.omega) * par.w * p
            return real_wage_net - mrs

        ell_a, u_a = np.nan, -np.inf
        if ell_kink < par.ell_max:
            try:
                res_a = root_scalar(foc_upper,
                                    bracket=[ell_kink, par.ell_max],
                                    method='brentq')
                if res_a.converged:
                    ell_a = res_a.root
                    u_a   = self.utility(self.post_tax_income(p, ell_a), ell_a)
            except ValueError:
                pass

        # ---- Step 4: global maximum ----
        options = [
            (u_b, ell_b, 'lower_bracket'),
            (u_k, ell_k, 'kink'),
            (u_a, ell_a, 'upper_bracket'),
        ]
        options.sort(key=lambda x: x[0], reverse=True)

        best_u, best_ell, best_type = options[0]
        return best_ell, best_u, best_type

    # ------------------------------------------------------------------
    # Population simulation
    # ------------------------------------------------------------------

    def simulate_population(self, N=10000, seed=12345):
        """
        Draw a population of N workers with log-normally distributed productivity.

        Parametrisation: log(p) ~ N(μ, σ²) with μ = −½σ² ensures E[p] = 1.
        This mean-normalisation is conventional so that the productivity
        distribution is centred at the representative agent.

        The log-normal is the empirically preferred distribution for earnings
        because it captures the right-skewed shape of observed income data.
        """
        par = self.par
        np.random.seed(seed)

        mu        = -0.5 * par.sigma_p ** 2   # mean-normalised log-normal
        par.p_vec = np.random.lognormal(mean=mu, sigma=par.sigma_p, size=N)
        return par.p_vec

    # ------------------------------------------------------------------
    # Social welfare function
    # ------------------------------------------------------------------

    def calculate_swf(self, p_vec=None):
        """
        Compute the Social Welfare Function (SWF) for a population.

        SWF = χ · G^η + Σ U_i

        where:
            G  = total tax revenue (= government expenditure on public goods)
            χ  = scaling weight on the public good
            η  = curvature of utility from the public good (0 < η < 1)
            U_i = individual utility of worker i

        The SWF embeds a trade-off: higher tax rates raise G and the public-
        good term, but reduce individual utilities through the labour-supply
        distortion.  The optimal tax design balances these forces.

        Returns
        -------
        (swf, total_tax, c_vec, ell_vec)
        """
        par = self.par
        if p_vec is None:
            p_vec = self.simulate_population()

        N       = len(p_vec)
        ell_vec = np.zeros(N)
        c_vec   = np.zeros(N)
        u_vec   = np.zeros(N)
        tax_vec = np.zeros(N)

        for i, p in enumerate(p_vec):
            ell, u, _ = self.optimal_choice_foc_kink(p)
            ell_vec[i] = ell
            u_vec[i]   = u
            income     = self.income(p, ell)
            tax        = self.tax(income)
            c_vec[i]   = income - tax
            tax_vec[i] = tax

        total_tax = np.sum(tax_vec)
        G         = total_tax   # government balances budget; all revenue → public good

        utility_from_G = (par.chi * (G ** par.eta) if G > 0 else -np.inf)
        swf            = utility_from_G + np.sum(u_vec)

        return swf, total_tax, c_vec, ell_vec

    # ------------------------------------------------------------------
    # Lorenz curve
    # ------------------------------------------------------------------

    def lorenz_curve(self, c_vec):
        """
        Compute Lorenz curve coordinates for a consumption distribution.

        The Lorenz curve plots the cumulative share of total consumption
        (y-axis) against the cumulative population share ranked by income
        (x-axis).  The further the curve lies below the 45° equality line,
        the more unequal the distribution — captured numerically by the Gini
        coefficient (= twice the area between the Lorenz curve and the diagonal).
        """
        c_sorted  = np.sort(c_vec)
        n         = len(c_sorted)
        cum_pop   = np.insert(np.arange(1, n + 1) / n, 0, 0)
        cum_c     = np.insert(np.cumsum(c_sorted) / np.sum(c_sorted), 0, 0)
        return cum_pop, cum_c
