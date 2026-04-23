"""
Problem 3: AS-AD Macroeconomic Model with Adaptive Expectations
===============================================================
Implements a discrete-time AS-AD (Aggregate Supply / Aggregate Demand) model
in the output-inflation space and simulates the economy's response to demand
shocks under adaptive inflation expectations.

Model equations
---------------
AD curve (from IS-MP framework):
    π_t = π* − (1/α) · [(y_t − ȳ) − z_t]
    where:  α  = b·a₁ / (1 + b·a₂)       (slope of IS multiplied by MP)
            z_t = v_t / (1 + b·a₂)        (demand shock pass-through)
            v_t = ρ·v_{t-1} + ε_t          (AR(1) demand shock process)

SRAS curve (short-run aggregate supply):
    π_t = π^e_t + γ · (y_t − ȳ)
    γ > 0 means firms set higher prices when output exceeds potential (ȳ).

Adaptive expectations:
    π^e_t = φ·π_{t-1} + (1−φ)·π^e_{t-1}
    0 ≤ φ ≤ 1 governs how quickly agents update beliefs.
    φ=1 → fully rational (static) expectations in the short run.
    φ=0 → expectations never update (extreme inertia).

Equilibrium
-----------
Each period equilibrium (y*_t, π*_t) is found analytically by solving the
AD and SRAS equations simultaneously, treating π^e_t and v_t as given.

Simulation
----------
The model is simulated for T periods.  A one-time demand shock (ε₁ > 0, ε_t=0 for t≥2)
shows how the economy converges back to the long-run equilibrium (ȳ, π*).
The speed of convergence depends on ρ (shock persistence) and φ (expectation rigidity).

Parameters (default calibration)
---------------------------------
    ȳ    = 1.0    potential output (normalised)
    π*   = 0.02   inflation target (2%)
    b    = 0.6    IS curve slope (investment sensitivity to interest rate)
    a₁   = 1.5    Taylor rule inflation coefficient
    a₂   = 0.10   Taylor rule output-gap coefficient
    γ    = 4.0    SRAS slope (price rigidity; higher γ → steeper SRAS)
    φ    = 0.6    expectation adjustment speed
"""

import numpy as np
import matplotlib.pyplot as plt


class ASADModelClass:

    def __init__(self, par=None):
    
        par = dict(
            ybar    = 1.0,
            pi_star = 0.02,
            b       = 0.6,
            a1      = 1.5,
            a2      = 0.10,
            gamma   = 4.0,
            phi     = 0.6
        )

        self.par = par.copy()

    def _alpha_z(self, v):

        p = self.par

        alpha = p['b'] * p['a1'] / (1.0 + p['b'] * p['a2'])
        z = v / (1.0 + p['b'] * p['a2'])

        return alpha, z

    def AD_curve(self, y, v):

        p = self.par
        alpha, z = self._alpha_z(v)
        inv_alpha = 1.0 / alpha
        return p['pi_star'] - inv_alpha * ((y - p['ybar']) - z)

    def SRAS_curve(self, y, pi_e):
        
        p = self.par
        return pi_e + p['gamma'] * (y - p['ybar'])

    # analytical equilibrium y_t^*, pi_t^* given pi_e and v
    def equilibrium(self, pi_e, v):
        p = self.par
        alpha, z = self._alpha_z(v)
        inv_alpha = 1.0 / alpha
        
        # Solve for y* at intersection of AD and SRAS
        y_star = p['ybar'] + (p['pi_star'] - pi_e + z * inv_alpha) / (p['gamma'] + inv_alpha)
        
        # Compute pi* using SRAS curve
        pi_star = pi_e + p['gamma'] * (y_star - p['ybar'])
        
        return y_star, pi_star

    # simulation
    def simulate(self, rho, eps):

        p = self.par
        T = len(eps)
        
        # Initialize arrays
        pi_e = np.zeros(T + 1)
        y_star = np.zeros(T + 1)
        pi_star = np.zeros(T + 1)
        v = np.zeros(T + 1)
        
        # Initial conditions
        pi_e[0] = p['pi_star']
        v[0] = 0
        
        # Simulate forward
        for t in range(T + 1):
            if t > 0:
                # Update expectations
                pi_e[t] = p['phi'] * pi_star[t-1] + (1 - p['phi']) * pi_e[t-1]
                
                # Compute demand shock (AR(1) process)
                v[t] = rho * v[t-1] + eps[t-1]
            
            # Compute equilibrium
            y_star[t], pi_star[t] = self.equilibrium(pi_e[t], v[t])
        
        return {
            'y': y_star,
            'pi': pi_star,
            'v': v,
            'pi_e': pi_e
        }

    # compute sd(y_gap), sd(pi), corr(y_gap, pi)
    def moments(self, y, pi):

        p = self.par
        y_gap = y - p['ybar']
        
        return {
            'sd_y_gap': np.std(y_gap),
            'sd_pi': np.std(pi),
            'corr': np.corrcoef(y_gap, pi)[0, 1]
        }
    
    # plot effect of inflation expectation jump
    def plot_expectation_jump(self, pi_e_initial=0.02, pi_e_jumped=0.08):
        
        p = self.par
        
        # Setting up data
        y_range = np.linspace(0.85, 1.15, 100)
        
        # Compute AD curve with v=0 
        pi_AD = self.AD_curve(y_range, v=0)
        
        # Compute SRAS curves
        pi_SRAS_initial = self.SRAS_curve(y_range, pi_e=pi_e_initial)
        pi_SRAS_shifted = self.SRAS_curve(y_range, pi_e=pi_e_jumped) 
        
        # Create figure and plot
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(y_range, pi_AD, 'b-', linewidth=2, label='AD curve')
        ax.plot(y_range, pi_SRAS_initial, 'r-', linewidth=2, label=r'SRAS ($\pi_t^e = \pi^* = 0.02$)')
        ax.plot(y_range, pi_SRAS_shifted, 'r--', linewidth=2, label=r'SRAS after jump ($\pi_t^e = 0.08$)')
        ax.plot(p['ybar'], p['pi_star'], 'ko', markersize=10, label=r'Long-run equilibrium ($y_t = \bar{y}, \pi_t = \pi^*$)')
        
        # Labels and title
        ax.set_xlabel(r'Output ($y_t$)', fontsize=12)
        ax.set_ylabel(r'Inflation ($\pi_t$)', fontsize=12)
        ax.set_title('AS-AD Model: Effect of Inflation Expectation Jump', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Show plots
        plt.tight_layout()
        plt.show()
        
        # Print the long-run equilibrium
        print(f"Long-run equilibrium: y = {p['ybar']}, π = {p['pi_star']}")
        print(f"When π^e_t jumps from {pi_e_initial} to {pi_e_jumped}, the SRAS curve shifts upward by {pi_e_jumped - pi_e_initial}")
    
    # plot simulation results
    def plot_simulation_results(self, results, rho_values, window=10):
        p = self.par
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        for i, rho in enumerate(rho_values):
            y = results[rho]['y']
            pi = results[rho]['pi']
            v = results[rho]['v']
            
            # Compute moving averages
            y_ma = np.convolve(y, np.ones(window)/window, mode='valid')
            pi_ma = np.convolve(pi, np.ones(window)/window, mode='valid')
            v_ma = np.convolve(v, np.ones(window)/window, mode='valid')
            
            # Plot output
            axes[0, i].plot(y, 'r-', linewidth=0.7, alpha=0.5, label='Output')
            axes[0, i].plot(range(window-1, len(y)), y_ma, 'r-', linewidth=2, label=f'{window}-period MA')
            axes[0, i].axhline(p['ybar'], color='gray', linestyle='--', alpha=0.7, label=r'$\bar{y}$')
            axes[0, i].set_ylabel(r'Output ($y_t$)', fontsize=11)
            axes[0, i].set_title(f'ρ = {rho}', fontsize=12, fontweight='bold')
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].legend(fontsize=9)
            
            # Plot inflation
            axes[1, i].plot(pi, 'orange', linewidth=0.7, alpha=0.5, label='Inflation')
            axes[1, i].plot(range(window-1, len(pi)), pi_ma, 'orange', linewidth=2, label=f'{window}-period MA')
            axes[1, i].axhline(p['pi_star'], color='gray', linestyle='--', alpha=0.7, label=r'$\pi^*$')
            axes[1, i].set_ylabel(r'Inflation ($\pi_t$)', fontsize=11)
            axes[1, i].grid(True, alpha=0.3)
            axes[1, i].legend(fontsize=9)
            
            # Plot demand shock
            axes[2, i].plot(v, 'purple', linewidth=0.7, alpha=0.5, label='Shock')
            axes[2, i].plot(range(window-1, len(v)), v_ma, 'purple', linewidth=2, label=f'{window}-period MA')
            axes[2, i].axhline(0, color='gray', linestyle='--', alpha=0.7)
            axes[2, i].set_xlabel('Time (t)', fontsize=11)
            axes[2, i].set_ylabel(r'Demand shock ($v_t$)', fontsize=11)
            axes[2, i].grid(True, alpha=0.3)
            axes[2, i].legend(fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def plot_period_dynamics(self, pi_e, v, y_star, pi_star):
        
        p = self.par
        T = len(pi_e) - 1
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        y_range = np.linspace(0.85, 1.15, 100)
        
        for t in range(T + 1):
            ax = axes[t]
            
            # Compute curves for this period
            pi_AD_t = self.AD_curve(y_range, v=v[t])
            pi_SRAS_t = self.SRAS_curve(y_range, pi_e=pi_e[t])
            
            # Plot
            ax.plot(y_range, pi_AD_t, 'b-', linewidth=2, label='AD curve')
            ax.plot(y_range, pi_SRAS_t, 'r-', linewidth=2, label=f'SRAS ($\\pi_t^e = {pi_e[t]:.4f}$)')
            ax.plot(y_star[t], pi_star[t], 'ko', markersize=8, label='Equilibrium')
            ax.axvline(p['ybar'], color='gray', linestyle='--', alpha=0.5)
            ax.axhline(p['pi_star'], color='gray', linestyle='--', alpha=0.5)
            
            ax.set_xlabel(r'Output ($y_t$)', fontsize=10)
            ax.set_ylabel(r'Inflation ($\pi_t$)', fontsize=10)
            ax.set_title(f'Period t = {t}', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 0.12])
        
        plt.tight_layout()
        plt.show()
        