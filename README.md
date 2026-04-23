# Nawid Rasekh | Economics Portfolio

BSc Economics student at the University of Copenhagen, currently in the
fourth semester. This repository collects my quantitative projects spanning applied economics, public policy, and computational methods.

I write production-quality Python, build models from first principles
rather than calling black-box solvers, and try to make every project
*reproducible*: public APIs over manual downloads, fixed seeds over
hand-tuned demos, and embedded notebook outputs so the results are visible
without cloning the repo.

---

## Projects

### [ev-road-pricing-denmark](./ev-road-pricing-denmark/)
**Road Pricing in Denmark's Electric Vehicle Transition: A Welfare Analysis of VKT Externalities**

Driver-level welfare model of five road-pricing policies as the Danish car
fleet electrifies (2026–2035). Heterogeneous price elasticities by income,
age, and location decompose the impact on consumer surplus, government
revenue, congestion externalities, and distributional equity. The proposed
hybrid policy captures **84% of the Pigouvian welfare gain** while raising
**9% more revenue** and avoiding the regressive dropout caused by a flat
annual fee.

`Python` · `pandas` · `NumPy` · `matplotlib` · `welfare economics` · `Pigouvian taxation` · `case competition`

### [danish-house-prices](./danish-house-prices/)
**Real vs. Nominal House Prices, Regional Divergence, and the Long Shadow of 2008**

CPI-deflated regional house price analysis (1992–present) using live
Danmarks Statistik APIs. Documents pronounced regional divergence
(Copenhagen vs. periphery, polarisation correlation r ≈ 0.6–0.7) and
shows that several peripheral municipalities are **still trading below
their pre-2008 real peak** more than a decade after the financial crisis.

`Python` · `pandas` · `dstapi` · `index deflation` · `regional economics`

### [labour-supply-kinked-tax](./labour-supply-kinked-tax/)
**Labour Supply with a Kinked Budget Constraint**

Computational labour-supply model with a piecewise-linear two-bracket tax
schedule. A four-step FOC solver handles the kink that breaks naive
optimisers, and a 10 000-worker population simulation shows that
**~40% of workers bunch at the kink**, matching the bunching patterns documented by Saez (2010) and Kleven & Waseem
(2013). The top tax improves social welfare and cuts the consumption Gini
by ~15% in the calibrated economy.

`Python` · `SciPy` · `Brent's method` · `welfare analysis` · `Lorenz/Gini`

### [inflation-cpi-hicp](./inflation-cpi-hicp/)
**CPI vs. HICP: International Inflation Comparison**

Tests whether Danish CPI and HICP can be used interchangeably for
cross-country comparisons (correlation = 0.9994), then benchmarks Danish
inflation against Austria, the Euro Area, and the United States across the
post-COVID inflation spike. Live API ingestion from Danmarks Statistik and
FRED.

`Python` · `pandas` · `dstapi` · `pandas-datareader` · `time-series alignment`

### [programming-for-economists](./programming-for-economists/)
**Group exam contributions: General Equilibrium and AS-AD**

Two exam-project problems where my groupmates wrote the primary
implementation: a Walrasian exchange economy with CES preferences (three
equilibria, basin-of-attraction analysis under tâtonnement vs. Newton-
Raphson) and an AS-AD model with adaptive expectations. Kept in the
portfolio for joint review and report context — see the folder for full
attribution.

`Python` · `general equilibrium` · `macro simulation` · `group exam project`

---

## Course context

The four projects above (excluding `ev-road-pricing-denmark`, which is a
case-competition project) are all from the *Programming for Economists*
course at the University of Copenhagen, spring 2025. All exam-project
problems were submitted as group work with [Kasper Vinther](https://github.com/) and
Mads Wittrup; primary authorship is noted in each subfolder's README.

---

## How to use this repository

Each project folder is self-contained: its own README, its own
`requirements.txt`, its own notebook with embedded outputs, and its own
exported figures. Pick any project and run:

```bash
cd <project-folder>
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

---

*Nawid Rasekh · [GitHub](https://github.com/NawidRasekh) · BSc Economics, University of Copenhagen*
