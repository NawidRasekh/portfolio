"""
Problem 2: Walrasian Exchange Economy with CES Preferences
===========================================================
Implements a two-good, two-agent pure exchange economy and finds the
competitive (Walrasian) equilibrium using two iterative algorithms.

Economic setup
--------------
Two agents (A and B) are each endowed with quantities of goods 1 and 2.
Both agents maximise a CES (Constant Elasticity of Substitution) utility
function subject to their budget constraint.

CES utility:  u(x₁, x₂) = (α·x₁^ρ + β·x₂^ρ)^(1/ρ)

    α, β  — preference weights for each good
    ρ     — substitution parameter; σ = 1/(1−ρ) is the elasticity of
            substitution between goods.  ρ→0 gives Cobb-Douglas; ρ<0 gives
            gross complements.

CES demand (derived from utility maximisation):
    x₁ = [α^σ · p₁^(1−σ) / (α^σ · p₁^(1−σ) + β^σ)] · m/p₁
    x₂ = [β^σ               / (α^σ · p₁^(1−σ) + β^σ)] · m

where m is the agent's income (value of endowment at market prices p₁, p₂=1).

Equilibrium condition
---------------------
Good 2 is the numeraire (p₂ = 1).  Walras's Law means we only need to clear
the market for good 1:
    excess demand:  e₁(p₁) = (x₁^A + x₁^B) − (ω₁^A + ω₁^B) = 0

Algorithms
----------
1. Tâtonnement (price adjustment):
       p₁_{k+1} = p₁_k + ν · e₁(p₁_k)
   Intuition: raise (lower) p₁ when there is excess demand (supply).
   Convergence is guaranteed for gross substitutes by Arrow-Hurwicz stability.

2. Newton-Raphson (dampened):
       p₁_{k+1} = p₁_k − φ · e₁(p₁_k) / [∂e₁/∂p₁]
   Uses a numerical derivative and a damping factor φ < 1 to prevent
   overshooting.  Much faster convergence than tâtonnement near the solution.

Multiple equilibria
--------------------
With the chosen CES parameters (ρ < 0 → complements) the economy can admit
three equilibria.  We map the basin of attraction — which starting price
converges to which equilibrium — by running the algorithm from 500 initial
conditions.
"""

from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt


class ExchangeEconomyModelClass:

    def __init__(self):
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        par.alpha_A = 1.0
        par.beta_A = (12/37)**3
        par.rho_A = -2.0
         
        par.alpha_B = (12/37)**3
        par.beta_B = 1.0
        par.rho_B = -2.0

        par.w1A = 1.0-1e-8
        par.w2A = 1e-8

        par.tol = 1e-8
        par.K = 5000
        
        par.nu = 50.0

        par.varphi = 0.1
        par.iota = 0.99

    def CES_utility(self,x1,x2,alpha,beta,rho):

        return (alpha*x1**rho + beta*x2**rho)**(1/rho)
    
    def CES_indifference(self,u,x1,alpha,beta,rho):

        x2 = np.nan*np.ones_like(x1)

        temp = (u**rho-alpha*x1**rho)/beta
        I = temp >= 0
        x2[I] = temp[I]**(1/rho)

        return x2

    def CES_demand(self,p1,m,alpha,beta,rho):

        sigma = 1/(1-rho)

        fac1 = alpha**sigma*p1**(1-sigma)
        fac2 = beta**sigma
    
        denom = fac1 + fac2
        x1 = fac1/p1*m/denom
        x2 = fac2*m/denom

        assert np.isclose(p1*x1+x2,m), 'budget constraint not satisfied'

        return x1, x2
    
    def utility_A(self,x1A,x2A):

        par = self.par
        return self.CES_utility(x1A,x2A,par.alpha_A,par.beta_A,par.rho_A)

    def x2A_indifference(self,uA,x1A):

        par = self.par
        return self.CES_indifference(uA,x1A,par.alpha_A,par.beta_A,par.rho_A)

    def utility_B(self,x1B,x2B):

        par = self.par
        return self.CES_utility(x1B,x2B,par.alpha_B,par.beta_B,par.rho_B)

    def x2B_indifference(self,uB,x1B):

        par = self.par
        return self.CES_indifference(uB,x1B,par.alpha_B,par.beta_B,par.rho_B)

    def demand_A(self,p1,m=None):

        if m is None: m = p1*self.par.w1A + self.par.w2A
        return self.CES_demand(p1,m,self.par.alpha_A,self.par.beta_A,self.par.rho_A)

    def demand_B(self,p1,m=None):

        if m is None: m = p1*(1-self.par.w1A) + (1-self.par.w2A)
        return self.CES_demand(p1,m,self.par.alpha_B,self.par.beta_B,self.par.rho_B)
    
    def create_edgeworthbox(self,figsize=(6,6)):

        par = self.par

        w1bar = 1.0
        w2bar = 1.0

        fig = plt.figure(figsize=figsize, dpi=100)
        ax_A = fig.add_subplot(1,1,1)

        ax_A.set_xlabel('$x_1^A$')
        ax_A.set_ylabel('$x_2^A$')

        temp = ax_A.twinx()
        temp.set_ylabel('$x_2^B$')

        ax_B = temp.twiny()
        ax_B.set_xlabel('$x_1^B$')
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()

        ax_A.plot([0,w1bar],[0,0],lw=2,color='black')
        ax_A.plot([0,w1bar],[w2bar,w2bar],lw=2,color='black')
        ax_A.plot([0,0],[0,w2bar],lw=2,color='black')
        ax_A.plot([w1bar,w1bar],[0,w2bar],lw=2,color='black')

        ax_A.set_xlim([-0.1, w1bar + 0.1])
        ax_A.set_ylim([-0.1, w2bar + 0.1])    
        ax_B.set_xlim([w1bar + 0.1, -0.1])
        ax_B.set_ylim([w2bar + 0.1, -0.1])

        ax_A.set_zorder(ax_B.get_zorder()+1)
        ax_A.patch.set_visible(False)

        return fig, ax_A, ax_B

    def indifference_curve_A(self,ax_A,x1A,x2A,**kwargs):

        uA = self.utility_A(x1A,x2A)
        
        x1A_grid = np.linspace(0.0001,0.9999,1000)
        x2A_grid = self.x2A_indifference(uA,x1A_grid)
        
        I = (x2A_grid > 0) & (x2A_grid < 1)
        ax_A.plot(x1A_grid[I],x2A_grid[I],**kwargs)

    def indifference_curve_B(self,ax_B,x1B,x2B,**kwargs):

        uB = self.utility_B(x1B,x2B)
        
        x1B_grid = np.linspace(0.0001,0.9999,1000)
        x2B_grid = self.x2B_indifference(uB,x1B_grid)
        
        I = (x2B_grid > 0) & (x2B_grid < 1)
        ax_B.plot(x1B_grid[I],x2B_grid[I],**kwargs)

    def plot_budget_line(self,ax_A):

        par = self.par
        sol = self.sol

        x1A_grid = np.linspace(0,1,100)
        x2_A = par.w2A-sol.p1*(x1A_grid-par.w1A)
        
        I = (x2_A > 0) & (x2_A < 1)
        ax_A.plot(x1A_grid[I],x2_A[I],color='black',ls='--',label='budget line')

    def add_legend(self,ax_A,ax_B,bbox_to_anchor=(0.10,0.60)):

        handles_A, labels_A = ax_A.get_legend_handles_labels()
        handles_B, labels_B = ax_B.get_legend_handles_labels()
        ax_A.legend(handles_A+handles_B,labels_A+labels_B,
                    bbox_to_anchor=bbox_to_anchor,loc='lower left',
                    facecolor='white',framealpha=0.90,fontsize=12)

    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = (x1A-par.w1A) + x1B-(1-par.w1A)
        eps2 = (x2A-par.w2A) + x2B-(1-par.w2A)

        return eps1,eps2


def tatonnement(model, p1_init, tau=1e-8, nu=0.8, K=5000, verbose=True):

    # Initialize tracking dictionary and set starting price
    history = {'p1': [p1_init], 'e1': [], 'iteration': []}
    p1_k = p1_init
    
    for k in range(K):
        # Calculate excess demand
        e1_k = model.check_market_clearing(p1_k)[0]
        history['e1'].append(e1_k)
        history['iteration'].append(k)

        # Check convergence
        if np.abs(e1_k) < tau:
            if verbose:
                print(f"  Converged at iteration {k}: "
                      f"p1* = {p1_k:.10f}, e1 = {e1_k:.2e}")
            return p1_k, history

        # Update price: p1_{k+1} = p1_k + nu * e1_k
        p1_k = p1_k + nu * e1_k
        history['p1'].append(p1_k)

    if verbose:
        print(f"  Did not converge after {K} iterations")
        print(f"  Final: p1 = {p1_k:.10f}, e1 = {history['e1'][-1]:.2e}")

    return p1_k, history


def newton_raphson_dampened(model, p1_init, tau=1e-8, varphi=0.1,
                             iota=0.99, h=1e-6, K=5000, verbose=True):
    history = {'p1': [p1_init], 'e1': [], 'iteration': [], 'Delta': []}
    p1_k = p1_init

    for k in range(K):
        # Calculate excess demand
        e1_k = model.check_market_clearing(p1_k)[0]
        history['e1'].append(e1_k)
        history['iteration'].append(k)

        # Check convergence
        if np.abs(e1_k) < tau:
            if verbose:
                print(f"  Converged at iteration {k}: "
                      f"p1* = {p1_k:.10f}, e1 = {e1_k:.2e}")
            return p1_k, history

        # Calculate numerical derivative
        e1_k_plus_h = model.check_market_clearing(p1_k + h)[0]
        Delta = (e1_k_plus_h - e1_k) / h
        history['Delta'].append(Delta)

        # Update price
        p1_k_new = p1_k - varphi * (e1_k / Delta)

        # Handle negative prices
        if p1_k_new < 0:
            p1_k = iota * p1_k
        else:
            p1_k = p1_k_new

        history['p1'].append(p1_k)

    if verbose:
        print(f"  Did not converge after {K} iterations")
        print(f"  Final: p1 = {p1_k:.10f}, e1 = {history['e1'][-1]:.2e}")

    return p1_k, history


def find_unique_equilibria(p1_final, tolerance=1e-4):
  
    unique_equilibria = []
    for p1 in p1_final:
        is_new = True
        for eq in unique_equilibria:
            if np.abs(p1 - eq) < tolerance:
                is_new = False
                break
        if is_new:
            unique_equilibria.append(p1)
    return unique_equilibria

def plot_demand_and_excess_demand(p1_grid, x1A_vec, x1B_vec, e1_vec, par, equilibrium_prices):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Demand for good 1
    ax1.plot(p1_grid, x1A_vec, label='$x_1^A(p_1, m^A)$', linewidth=2)
    ax1.plot(p1_grid, x1B_vec, label='$x_1^B(p_1, m^B)$', linewidth=2)
    ax1.axhline(y=par.w1A, color='red', linestyle='--', alpha=0.5, label='$\\omega_1^A$ (endowment A)')
    ax1.axhline(y=1-par.w1A, color='orange', linestyle='--', alpha=0.5, label='$\\omega_1^B$ (endowment B)')
    ax1.set_xlabel('$p_1$', fontsize=12)
    ax1.set_ylabel('Quantity demanded', fontsize=12)
    ax1.set_title('Question 2.1: Demand for Good 1', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right plot: Excess demand with sign changes marked
    ax2.plot(p1_grid, e1_vec, label='$e_1(p_1)$', linewidth=2.5, color='green')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.fill_between(p1_grid, e1_vec, 0, alpha=0.15, color='green')
    # Mark equilibria
    for i, p_eq in enumerate(equilibrium_prices):
        idx = np.argmin(np.abs(p1_grid - p_eq))
        ax2.plot(p_eq, e1_vec[idx], '*', markersize=25, zorder=5, 
                 label=f'Eq {i+1}: p*={p_eq:.3f}')
    ax2.set_xlabel('$p_1$', fontsize=12)
    ax2.set_ylabel('Excess demand $e_1(p_1)$', fontsize=12)
    ax2.set_title('Question 2.1: Excess Demand for Good 1', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_tatonnement_convergence(histories, p1_stars, labels):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['blue', 'green', 'orange']
    markers = ['o-', '^-', 's-']
    
    for i, (history, p1_star, label) in enumerate(zip(histories, p1_stars, labels)):
        min_len = len(history['iteration'])
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # Left plot: Prices over iterations
        ax1.plot(history['iteration'], history['p1'][:min_len], marker,
                 label=f'$p_1^0 = {label}$ → $p_1^* = {p1_star:.3f}$', 
                 markersize=3, linewidth=1.5, color=color)
        ax1.axhline(y=p1_star, color=color, linestyle='--', alpha=0.3, linewidth=2)
        
        # Right plot: Excess demand over iterations
        ax2.plot(history['iteration'], history['e1'], marker,
                 label=f'$p_1^0 = {label}$', markersize=3, linewidth=1.5, color=color)
    
    ax1.set_xlabel('Iteration $k$', fontsize=12)
    ax1.set_ylabel('Price $p_1^k$', fontsize=12)
    ax1.set_title('Question 2.2: Tâtonnement - Price Convergence', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5, label='Market clearing (e₁=0)')
    ax2.set_xlabel('Iteration $k$', fontsize=12)
    ax2.set_ylabel('Excess demand $e_1^k$', fontsize=12)
    ax2.set_title('Question 2.2: Tâtonnement - Excess Demand', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('symlog', linthresh=1e-8)

    plt.tight_layout()
    plt.show()


def plot_basin_of_attraction(p1_init_grid, p1_final, unique_equilibria, title='Basin of Attraction'):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(p1_init_grid, p1_final, 'o-', linewidth=2, markersize=4, label='Equilibrium $p_1^*$')
    ax.plot(p1_init_grid, p1_init_grid, '--', color='red', alpha=0.5, linewidth=1.5, label='45° line (identity)')

    # Mark unique equilibria
    for eq in sorted(unique_equilibria):
        ax.plot(eq, eq, 'r*', markersize=20, zorder=5)

    ax.set_xlabel('Initial guess $p_1^0$', fontsize=12)
    ax.set_ylabel('Equilibrium $p_1^*$', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_edgeworth_box_equilibria(model, par, allocations, equilibrium_labels, colors):
    fig, ax_A, ax_B = model.create_edgeworthbox()

    # Plot endowments
    ax_A.plot(par.w1A, par.w2A, 'ko', markersize=10, label='Endowment W', zorder=5)

    # Plot all equilibria
    for i, alloc in enumerate(allocations):
        # Plot indifference curves
        model.indifference_curve_A(ax_A, alloc['x1A'], alloc['x2A'], 
                                  color=colors[i], linewidth=1.5, alpha=0.7)
        model.indifference_curve_B(ax_B, alloc['x1B'], alloc['x2B'], 
                                  color=colors[i], linewidth=1.5, alpha=0.7)
        
        # Plot budget lines
        x1A_grid = np.linspace(0, 1, 100)
        x2_A = par.w2A - alloc['p1'] * (x1A_grid - par.w1A)
        I = (x2_A > 0) & (x2_A < 1)
        ax_A.plot(x1A_grid[I], x2_A[I], color=colors[i], ls='--', alpha=0.5, linewidth=1)
        
        # Plot equilibrium points
        ax_A.plot(alloc['x1A'], alloc['x2A'], '*', markersize=18, 
                 color=colors[i], label=f'E{i+1} {equilibrium_labels[i]}', zorder=5)
        
        # Add labels with equilibrium number
        offset_x = 0.02 if i != 1 else -0.08
        offset_y = 0.02 if i == 0 else (-0.05 if i == 1 else 0.02)
        ax_A.text(alloc['x1A'] + offset_x, alloc['x2A'] + offset_y, 
                 f'E{i+1}', fontsize=9, fontweight='bold', color=colors[i])

    ax_A.set_title('Question 2.5: All Three Equilibria in Edgeworth Box', fontsize=13, fontweight='bold')
    model.add_legend(ax_A, ax_B)

    plt.tight_layout()
    plt.show()
