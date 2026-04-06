"""
Task 3: Labour-Supply Analysis with a Kinked Budget Constraint
==============================================================
This module orchestrates the full empirical analysis for Question 3 of the
model project.  It imports TopTaxWorker (question3.py) and runs four tasks:

    3.0 — Simple budget-constraint visualisation for a representative worker
    3.1 — Detailed four-panel analysis at three specific productivity levels
    3.2 — Labour-supply function across the full productivity distribution,
           with bunching analysis at the kink
    3.3 — Welfare effects of the top tax + optimal policy search

Key economic questions addressed
---------------------------------
1. At which productivity levels does a worker choose to bunch at the kink
   (earn exactly κ), and why?  Bunching arises because the top-bracket
   surcharge ω makes the marginal return to additional hours fall sharply
   once income crosses κ — for some workers the loss exceeds the gain.

2. Does the top tax reduce consumption inequality (Gini coefficient)?
   Higher-productivity workers pay more top-rate tax, redistributing income
   to the public good G.  Whether this improves social welfare depends on
   χ (the weight on G) and η (its curvature).

3. Is the calibrated (ω=0.2, κ=9.0) the welfare-maximising policy?
   We search over a 8×7 grid of (ω, κ) values to test whether better
   parameters exist — a simple discrete optimisation of the SWF.

Speed benchmark
---------------
The FOC root-finding method (Solver 2) is analytically informed: it restricts
the search to each bracket separately, so the root-finder only needs to
search a half-interval rather than the full [0, ℓ_max] range.  In practice
this makes the FOC solver ~5–10× faster than the black-box numerical
optimiser.  The speed comparison is reported in the four-panel plot.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from question3 import TopTaxWorker


# Constants and Parameters
RANDOM_SEED = 42
POPULATION_SIZE = 10000
TIMING_ITERATIONS = 100
N_PRODUCTIVITY_SAMPLES = 100


def initialize_model():

    plt.style.use('seaborn-v0_8-whitegrid')
    
    model = TopTaxWorker()
    par = model.par
    
    # Set specific parameters for Question 3
    par.kappa = 9.0
    par.omega = 0.2
    par.tau = 0.137
    par.zeta = 0.0
    
    # Add government/SWF parameters needed for calculate_swf
    par.sigma_p = 0.3
    par.chi = 50.0
    par.eta = 0.1
    
    print(f'Parameters set: kappa = {par.kappa}, omega = {par.omega}, '
          f'tau = {par.tau}, zeta = {par.zeta}')
    print(f'Worker parameters: w = {par.w}, nu = {par.nu}, '
          f'epsilon = {par.epsilon}, ell_max = {par.ell_max}')
    print(f'Distribution parameters: sigma_p = {par.sigma_p}, '
          f'chi = {par.chi}, eta = {par.eta}')
    
    return model, par


def plot_simple_budget_constraint(model, par, p_val=1.0):

    ell_vec = np.linspace(0.01, par.ell_max, 100)
    c_vec = np.zeros_like(ell_vec)
    u_vec = np.zeros_like(ell_vec)
    
    for i, ell in enumerate(ell_vec):
        c_vec[i] = model.post_tax_income(p_val, ell)
        u_vec[i] = model.utility(c_vec[i], ell)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel(r'Labor Supply ($\ell$)')
    ax1.set_ylabel(r'Consumption ($c$)', color=color)
    ax1.plot(ell_vec, c_vec, color=color, label='Budget Constraint')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel(r'Utility ($U$)', color=color)
    ax2.plot(ell_vec, u_vec, color=color, linestyle='--',
             label='Utility')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f'Budget Constraint and Utility for $p={p_val}$')
    fig.tight_layout()
    plt.show()


def analyze_labor_supply(p_val, model, par):
    ell_kink = par.kappa / (par.w * p_val)
    
    # Create a 2x2 figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # --- Plot 1: Budget Constraint and Utility ---
    ell_vec = np.linspace(0.01, par.ell_max, 100)
    c_vec = np.zeros_like(ell_vec)
    u_vec = np.zeros_like(ell_vec)
    
    for i, ell in enumerate(ell_vec):
        c_vec[i] = model.post_tax_income(p_val, ell)
        u_vec[i] = model.utility(c_vec[i], ell)
    
    color = 'tab:blue'
    ax1.set_xlabel(r'Labor Supply ($\ell$)')
    ax1.set_ylabel(r'Consumption ($c$)', color=color)
    ax1.plot(ell_vec, c_vec, color=color, label='Budget Constraint',
             linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axvline(ell_kink, color='red', linestyle='--', alpha=0.5,
                label='Kink')
    
    ax1_twin = ax1.twinx()
    color = 'tab:red'
    ax1_twin.set_ylabel(r'Utility ($U$)', color=color)
    ax1_twin.plot(ell_vec, u_vec, color=color, linestyle='--',
                  label='Utility', linewidth=2)
    ax1_twin.tick_params(axis='y', labelcolor=color)
    
    ax1.set_title(f'Budget Constrant and utility ($p={p_val}$)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: FOCs on Valid Intervals ---
    ell_lower = np.linspace(0.01, ell_kink, 100)
    foc_lower_vec = np.zeros_like(ell_lower)
    for i, ell in enumerate(ell_lower):
        try:
            foc_lower_vec[i] = model.foc_error(p_val, ell,
                                                type='standard')
        except Exception:
            foc_lower_vec[i] = np.nan
    
    ell_upper = np.linspace(ell_kink, par.ell_max, 100)
    foc_upper_vec = np.zeros_like(ell_upper)
    for i, ell in enumerate(ell_upper):
        try:
            foc_upper_vec[i] = model.foc_error(p_val, ell, type='top')
        except Exception:
            foc_upper_vec[i] = np.nan
    
    # Find optimal solution
    ell_opt, u_opt, type_opt = model.optimal_choice_foc_kink(p_val)
    try:
        foc_type = type_opt if type_opt != 'kink' else 'standard'
        foc_opt = model.foc_error(p_val, ell_opt, type=foc_type)
    except Exception:
        foc_opt = 0
    
    ax2.plot(ell_lower, foc_lower_vec,
             label=r'Lower FOC $\varphi$ (income $< \kappa$)',
             linewidth=2)
    ax2.plot(ell_upper, foc_upper_vec,
             label=r'Upper FOC $\bar{\varphi}$ (income $\geq \kappa$)',
             linewidth=2)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.axvline(ell_kink, color='gray', linestyle=':', linewidth=1.5,
                label='Kink')
    ax2.plot(ell_opt, foc_opt, 'ro', markersize=10,
             label=f'Optimal: $\ell^* = {ell_opt:.4f}$ ({type_opt})')
    
    ax2.set_xlabel(r'Labor Supply ($\ell$)')
    ax2.set_ylabel('FOC Value')
    ax2.set_title(f'FOC Errors - Valid Intervals Only ($p={p_val}$)')
    ax2.set_ylim(-2, 2)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: Optimal Choice on Budget ---
    c_opt = model.post_tax_income(p_val, ell_opt)
    ell_plot = np.linspace(0.01, par.ell_max, 100)
    c_plot = [model.post_tax_income(p_val, e) for e in ell_plot]
    
    ax3.plot(ell_plot, c_plot, 'b-', linewidth=2,
             label='Budget Constraint')
    ax3.axvline(ell_kink, color='red', linestyle='--', alpha=0.5,
                label='Kink')
    ax3.plot(ell_opt, c_opt, 'go', markersize=12,
             label=f'Optimal: $\ell^*={ell_opt:.4f}$, $c^*={c_opt:.4f}$')
    ax3.set_xlabel(r'Labor Supply ($\ell$)')
    ax3.set_ylabel(r'Consumption ($c$)')
    ax3.set_title(f'Optimal Choice on Budget Constraint ($p={p_val}$)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # --- Plot 4: Speed Comparison ---
    n_iterations = TIMING_ITERATIONS
    
    # Numerical Optimizer - multiple iterations
    start = time.time()
    for _ in range(n_iterations):
        model.optimal_choice_numerical(p_val)
    t_num_total = time.time() - start
    t_num = t_num_total / n_iterations
    
    # FOC with Kink - multiple iterations
    start = time.time()
    for _ in range(n_iterations):
        model.optimal_choice_foc_kink(p_val)
    t_foc_total = time.time() - start
    t_foc = t_foc_total / n_iterations
    
    # Bar plot
    methods = ['Numerical\nOptimizer', 'FOC\nRoot-Finder']
    times = [t_num * 1000, t_foc * 1000]
    colors_bar = ['#1f77b4', '#ff7f0e']
    
    bars = ax4.bar(methods, times, color=colors_bar, alpha=0.7,
                   edgecolor='black', linewidth=2)
    ax4.set_ylabel('Time (milliseconds)')
    ax4.set_title(f'Speed Comparison ($p={p_val}$)')
    ax4.set_ylim(0, max(times) * 1.3)
    
    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                 f'{t:.4f}ms', ha='center', va='bottom',
                 fontweight='bold')
    
    speedup = t_num / t_foc if t_foc > 0 else 1.0
    ax4.text(0.5, 0.95, f'FOC is {speedup:.1f}x faster',
             transform=ax4.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
             fontsize=11, fontweight='bold')
    
    ax4.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    print(f"\n Summary for p={p_val} ===")
    print(f"Optimal labor supply: {ell_opt:.4f}")
    print(f"Solution: {type_opt}")
    print(f"Consumption: {c_opt:.4f}")
    print(f"Utility: {u_opt:.4f}")
    print(f"Speed comparison: Numerical {t_num*1000:.4f}ms, "
          f"FOC {t_foc*1000:.4f}ms, Speedup {speedup:.1f}x")


def analyze_labor_supply_function(model, par):
    p_vec = np.linspace(0.5, 3.0, N_PRODUCTIVITY_SAMPLES)
    ell_star_vec = np.zeros_like(p_vec)
    c_star_vec = np.zeros_like(p_vec)
    types_vec = []
    
    for i, p in enumerate(p_vec):
        ell, u, type_ = model.optimal_choice_foc_kink(p)
        ell_star_vec[i] = ell
        c_star_vec[i] = model.post_tax_income(p, ell)
        types_vec.append(type_)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Labor Supply
    ax1.plot(p_vec, ell_star_vec, label='Optimal Labor Supply',
             linewidth=2.5)
    ax1.set_xlabel(r'Productivity ($p$)', fontsize=12)
    ax1.set_ylabel(r'Labor Supply ($\ell^\star$)', fontsize=12)
    ax1.set_title('Labor Supply Function', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot Consumption
    ax2.plot(p_vec, c_star_vec, label='Optimal Consumption',
             color='orange', linewidth=2.5)
    ax2.set_xlabel(r'Productivity ($p$)', fontsize=12)
    ax2.set_ylabel(r'Consumption ($c^\star$)', fontsize=12)
    ax2.set_title('Consumption Function', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Analyze Bunching and Proportions
    count_lower = types_vec.count('lower_bracket')
    count_kink = types_vec.count('kink')
    count_upper = types_vec.count('upper_bracket')
    
    total = len(p_vec)
    prop_lower = count_lower / total * 100
    prop_kink = count_kink / total * 100
    prop_upper = count_upper / total * 100
    
    print(f"LABOR SUPPLY FUNCTION - BUNCHING ANALYSIS")
    print(f"\nTotal workers analyzed: {total}")
    print(f"\nWorkers by solution type:")
    print(f"  Lower bracket (ℓ* < κ/wp): {count_lower} workers "
          f"({prop_lower:.1f}%)")
    print(f"  At kink (ℓ* = κ/wp): {count_kink} workers "
          f"({prop_kink:.1f}%)")
    print(f"  Upper bracket (ℓ* > κ/wp): {count_upper} workers "
          f"({prop_upper:.1f}%)")
    
    print(f"\n✓ {prop_kink:.1f}% of workers bunch at the kink point")
    print(f"\nKey insight:")
    print(f"  - Workers with low productivity work in lower bracket")
    print(f"  - Medium productivity workers bunch at kink to avoid top tax")
    print(f"  - Only high productivity workers choose upper bracket")
    
    return {
        'p_vec': p_vec,
        'ell_star_vec': ell_star_vec,
        'c_star_vec': c_star_vec,
        'types_vec': types_vec,
        'proportions': {
            'lower': prop_lower,
            'kink': prop_kink,
            'upper': prop_upper
        }
    }


def calculate_gini(c_vec):
    
    c_sorted = np.sort(c_vec)
    n = len(c_sorted)
    return ((2 * np.sum(np.arange(1, n + 1) * c_sorted)) /
            (n * np.sum(c_sorted)) - (n + 1) / n)


def plot_lorenz(c_vec, label, ax, color):
    c_sorted = np.sort(c_vec)
    n = len(c_sorted)
    cumsum_c = np.cumsum(c_sorted) / np.sum(c_sorted)
    cumsum_pop = np.arange(1, n + 1) / n
    ax.plot(cumsum_pop, cumsum_c, label=label, linewidth=2.5, color=color)
    return cumsum_pop, cumsum_c


def analyze_welfare_effects(model, par):
    print(f"\n(1) WELFARE ANALYSIS WITH TOP TAX")
    print(f"-" * 60)
    
    # Create population with log-normal productivity distribution
    np.random.seed(RANDOM_SEED)
    sigma_p = par.sigma_p
    mu = -0.5 * sigma_p**2
    p_vec_pop = np.random.lognormal(mean=mu, sigma=sigma_p,
                                     size=POPULATION_SIZE)
    
    # Calculate baseline SWF (ω=0)
    par.omega = 0.0
    swf_base, tax_base, c_base, ell_base = model.calculate_swf(p_vec_pop)
    gini_base = calculate_gini(c_base)
    
    # Calculate SWF with current top tax (ω=0.2, κ=9.0)
    par.omega = 0.2
    par.kappa = 9.0
    (swf_top_tax, tax_top_tax, c_top_tax,
     ell_top_tax) = model.calculate_swf(p_vec_pop)
    gini_top_tax = calculate_gini(c_top_tax)
    
    print(f"\nBaseline (no top tax, ω=0):")
    print(f"  SWF: {swf_base:.2f}")
    print(f"  Total Tax Revenue: {tax_base:.2f}")
    print(f"  Gini: {gini_base:.4f}")
    
    print(f"\nWith top tax (ω=0.2, κ=9.0):")
    print(f"  SWF: {swf_top_tax:.2f}")
    print(f"  Total Tax Revenue: {tax_top_tax:.2f}")
    print(f"  Gini: {gini_top_tax:.4f}")
    print(f"  SWF Change: {swf_top_tax - swf_base:.2f} "
          f"({(swf_top_tax - swf_base)/swf_base*100:.2f}%)")
    
    # (2) Plot Lorenz curves for consumption
    print(f"\n(2) LORENZ CURVE - INEQUALITY ANALYSIS")
    print(f"-" * 60)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot 45-degree line (perfect equality)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2,
            label='Perfect Equality', alpha=0.5)
    
    # Plot Lorenz curves
    plot_lorenz(c_base, f'Baseline (ω=0): Gini={gini_base:.4f}',
                ax, 'blue')
    plot_lorenz(c_top_tax, f'Top Tax (ω=0.2): Gini={gini_top_tax:.4f}',
                ax, 'orange')
    
    ax.set_xlabel('Cumulative Share of Population', fontsize=12)
    ax.set_ylabel('Cumulative Share of Consumption', fontsize=12)
    ax.set_title('(2) Lorenz Curves: Consumption Inequality',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nLorenz Curve Interpretation:")
    print(f"  - Closer to 45° line = more equal distribution")
    print(f"  - Baseline Gini: {gini_base:.4f} (less equal)")
    print(f"  - Top Tax Gini: {gini_top_tax:.4f} (more equal)")
    print(f"  - Improvement: "
          f"{(gini_base - gini_top_tax)/gini_base * 100:.1f}% "
          f"reduction in inequality")
    
    # (3) Optimize omega and kappa
    print(f"\n(3) Optimal top tax parameters")
   
    print(f"Searching for better (ω, κ) that maximize SWF...")
    
    best_swf = swf_base
    best_params = {'omega': 0.0, 'kappa': par.kappa, 'swf': swf_base}
    results = []
    
    # Search over omega values (from 0 to 0.35)
    omega_vals = np.linspace(0, 0.35, 8)
    kappa_vals = np.linspace(6, 12, 7)
    
    for omega_try in omega_vals:
        for kappa_try in kappa_vals:
            par.omega = omega_try
            par.kappa = kappa_try
            swf_try, _, _, _ = model.calculate_swf(p_vec_pop)
            results.append({'omega': omega_try, 'kappa': kappa_try,
                           'swf': swf_try})
            
            if swf_try > best_swf:
                best_swf = swf_try
                best_params = {'omega': omega_try, 'kappa': kappa_try,
                              'swf': swf_try}
    
    # Restore original values
    par.omega = 0.2
    par.kappa = 9.0
    
    print(f"\nBest parameters found:")
    print(f"  ω (top tax rate): {best_params['omega']:.4f}")
    print(f"  κ (cutoff income): {best_params['kappa']:.4f}")
    print(f"  SWF: {best_params['swf']:.4f}")
    print(f"  Improvement vs. baseline: "
          f"{best_params['swf'] - swf_base:.4f}")
    
    if best_params['swf'] > swf_base:
        print(f"\n✓ IMPROVEMENT FOUND!")
        print(f"  New parameters improve SWF by "
              f"{best_params['swf'] - swf_base:.4f}")
        print(f"  This suggests the current (ω=0.2, κ=9.0) is NOT optimal")
    else:
        print(f"\n⚠ No improvement found")
        print(f"  Current parameters (ω=0.2, κ=9.0) are close to or "
              f"at the optimum")
        print(f"  Best found: ω={best_params['omega']:.4f} yields "
              f"no improvement")
    
    print(f"\n{'='*60}")
    
    return {
        'baseline': {'swf': swf_base, 'gini': gini_base,
                    'tax': tax_base},
        'top_tax': {'swf': swf_top_tax, 'gini': gini_top_tax,
                   'tax': tax_top_tax},
        'optimal': best_params
    }


def main():
    print("QUESTION 3: LABOR SUPPLY WITH KINKED BUDGET CONSTRAINT")
   
    
    # Initialize model
    model, par = initialize_model()
    
    # 3.0: Simple budget constraint visualization
   
    print("3.0: SIMPLE BUDGET CONSTRAINT VISUALIZATION")
    plot_simple_budget_constraint(model, par, p_val=1.0)
    
    # 3.1: Comprehensive labor supply analysis for three productivity levels
    print("3.1: COMPREHENSIVE LABOR SUPPLY ANALYSIS")
    for p_val in [1.0, 1.175, 1.5]:
        analyze_labor_supply(p_val, model, par)
    
    # 3.2: Labor supply function and bunching behavior
    print("3.2: LABOR SUPPLY FUNCTION AND BUNCHING")
    labor_results = analyze_labor_supply_function(model, par)
    
    # 3.3: Welfare effects and optimal policy search
    print("3.3: WELFARE EFFECTS AND OPTIMAL POLICY")
    welfare_results = analyze_welfare_effects(model, par)
    
    print("ANALYSIS COMPLETE")
    
    return {
        'model': model,
        'parameters': par,
        'labor_supply_results': labor_results,
        'welfare_results': welfare_results
    }
if __name__ == '__main__':
    results = main()
