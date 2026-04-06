"""
case_analysis.py  –  Core model library for PCC 2026
=====================================================
Reusable functions for loading data, estimating heterogeneous price
elasticities, and simulating road-pricing policy scenarios.

All monetary values are in DKK (Danish Krone). VKT = Vehicle Kilometres
Travelled. Fleet-level results are scaled from the sample (~2 000 drivers)
to the national fleet using the omega weights derived from Sheet B.

Usage (standalone sanity check):
    python case_analysis.py
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── File paths ────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
DATA_ZIP    = ROOT / "data" / "PCC2026.zip"
EXTRACT_DIR = ROOT / "data" / "pcc2026"
XLSX_PATH   = EXTRACT_DIR / "Dataset.xlsx"

# ── Segment definitions ───────────────────────────────────────────────────────
# Four driving contexts formed by crossing zone (city / countryside) with
# time of day (peak / off-peak). These are the unit of analysis for both
# externality costs and price elasticities.
SEGMENTS = ['city-peak', 'city-offpeak', 'countryside-peak', 'countryside-offpeak']

SEGMENT_ATTRS = {
    'city-peak':          {'zone': 'City',        'time': 'Peak',     'area': 'city'},
    'city-offpeak':       {'zone': 'City',        'time': 'Off-peak', 'area': 'city'},
    'countryside-peak':   {'zone': 'Countryside', 'time': 'Peak',     'area': 'countryside'},
    'countryside-offpeak':{'zone': 'Countryside', 'time': 'Off-peak', 'area': 'countryside'},
}

# ── Internal (private) driving costs – DKK per km (Sheet C, Table 1) ─────────
# These form the driver's baseline per-km price p0 before any road pricing.
# Fuel, tyre wear, repairs, depreciation, and (for EVs) battery replacement.
INTERNAL_COSTS = {
    'Fossil':      {'Fuel': 0.382, 'Tires': 0.072, 'Repairs': 0.37,  'Depreciation': 0.10,  'Battery': 0.00},
    'Electricity': {'Fuel': 0.177, 'Tires': 0.066, 'Repairs': 0.203, 'Depreciation': 0.303, 'Battery': 0.121},
}

# Aggregate p0 per fuel type (scalar used in elasticity calculations)
P0 = {k: sum(v.values()) for k, v in INTERNAL_COSTS.items()}
# Fossil: 0.924 DKK/km  |  Electricity: 0.870 DKK/km

# ── External costs – DKK per km (Sheet D) ────────────────────────────────────
# Marginal social costs beyond what the driver pays: congestion (the largest
# component in city-peak), accidents, noise, and road wear.
# City-peak congestion (2.40 DKK/km) is ~20× the countryside-offpeak value,
# which motivates differentiated road pricing.
OTHER_EXTERNAL = {
    'city-peak':          {'Congestion': 2.402, 'Accidents': 0.594, 'Noise': 0.154, 'Infrastructure': 0.011},
    'city-offpeak':       {'Congestion': 0.424, 'Accidents': 0.486, 'Noise': 0.126, 'Infrastructure': 0.009},
    'countryside-peak':   {'Congestion': 0.266, 'Accidents': 0.211, 'Noise': 0.021, 'Infrastructure': 0.010},
    'countryside-offpeak':{'Congestion': 0.042, 'Accidents': 0.152, 'Noise': 0.012, 'Infrastructure': 0.008},
}
OTHER_SUM = {seg: sum(d.values()) for seg, d in OTHER_EXTERNAL.items()}

# Pollution (air quality) externality – DKK per km (Sheet D)
# EVs still impose a small city-area cost due to particulate matter from
# brakes and tyres, even though tailpipe emissions are zero.
POLLUTION = {
    ('Fossil',      'city'):        0.15,
    ('Fossil',      'countryside'): 0.05,
    ('Electricity', 'city'):        0.01,
    ('Electricity', 'countryside'): 0.01,
}

# ── Elasticity parameters (Technical Appendix, Table 1) ──────────────────────
# Heterogeneous price elasticity specification from Eq. (24):
#
#   ε_{i,s} = γ₀
#             + β_low  · 1[low income]
#             + β_high · 1[high income]
#             + β_young · 1[young]  +  β_old · 1[old]
#             + β_urban · 1[home in city]
#             + δ_peak · 1[peak segment]
#             + δ_city · 1[city zone]
#             + interaction terms
#
# All parameters are negative or zero (demand falls when price rises).
# Base elasticity γ₀ = -0.40 (medium income, middle-aged, rural, off-peak).

PARAMS_FULL = {
    'gamma0':           -0.40,   # baseline elasticity
    'beta_low':         -0.12,   # low-income drivers respond more to price
    'beta_high':         0.06,   # high-income drivers respond less
    'beta_young':       -0.03,   # young drivers slightly more elastic
    'beta_old':          0.03,   # older drivers slightly less elastic
    'beta_urban':       -0.03,   # city residents more elastic (alternatives available)
    'delta_peak':        0.10,   # peak trips are less price-sensitive (necessity)
    'delta_city':       -0.03,   # city driving somewhat more elastic
    'beta_low_offpeak': -0.08,   # low income × off-peak interaction
    'beta_high_peak':    0.04,   # high income × peak interaction
    'delta_city_peak':   0.06,   # city-peak specific adjustment
    'delta_urban_city': -0.05,   # urban resident × city zone interaction
}

# Simplified model: income and segment effects only (no age / home location).
# Used for robustness checks reported alongside the full model.
PARAMS_INCOME_ONLY = {k: v for k, v in PARAMS_FULL.items()
                      if k in ('gamma0', 'beta_low', 'beta_high', 'delta_peak',
                               'delta_city', 'beta_low_offpeak', 'beta_high_peak',
                               'delta_city_peak')}

# Congestion externality curvature parameter (η in Technical Appendix).
# A value < 1 means congestion savings are concave in VKT reduction
# (each incremental km removed yields diminishing congestion relief).
ETA = 0.95


# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def ensure_data() -> None:
    """Extract the case ZIP archive if the Excel file is not yet present."""
    if XLSX_PATH.exists():
        return
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DATA_ZIP, "r") as zf:
        zf.extractall(EXTRACT_DIR)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare all three core datasets from Dataset.xlsx.

    Returns
    -------
    A : pd.DataFrame
        Long-format panel: one row per (driver, segment) observation.
        Columns include id, car, zone, time_of_day, distance_yearly, etc.

    drivers : pd.DataFrame
        Wide-format driver table indexed by driver id.
        Each row is one driver; segment VKT columns are pivoted wide.
        Also contains demographic variables (income, age, home location).

    fleet : pd.DataFrame
        Annual fleet composition 2026–2035 (Sheet B).
        Columns: year, fossil, ev, new_fossil, new_ev, total, ev_share_fleet,
        ev_share_new. Used to scale sample results to national level.
    """
    ensure_data()

    # ── Sheet A: driver–segment observations ─────────────────────────────────
    A = pd.read_excel(XLSX_PATH, sheet_name="Sheet A", header=13)
    A = A[A["id"].notna()].copy()
    A["id"] = A["id"].astype(int)
    A["distance_yearly"] = pd.to_numeric(A["distance_yearly"], errors="coerce").fillna(0.0)

    # Construct the four-way segment key used throughout the model
    A["segment"] = (A["zone"].str.lower() + "-"
                    + A["time_of_day"].str.lower()
                    .str.replace("off-peak", "offpeak")
                    .str.replace("peak", "peak"))

    # Build wide driver table: demographics + one VKT column per segment
    drivers = (A[["id", "car", "home_location", "income_mapped", "age_group_mapped"]]
               .drop_duplicates()
               .set_index("id"))
    vkt_wide = A.pivot_table(index="id", columns="segment",
                              values="distance_yearly", aggfunc="sum", fill_value=0)
    drivers = drivers.join(vkt_wide)

    # Attach baseline per-km cost so elasticity calculations have p0 available
    drivers["p0"] = drivers["car"].map(P0)

    # ── Sheet B: fleet composition projections ────────────────────────────────
    B = pd.read_excel(XLSX_PATH, sheet_name="Sheet B", header=None)
    years = [int(y) for y in B.iloc[14, 2:].tolist()]
    fleet = pd.DataFrame({
        "year":       years,
        "fossil":     B.iloc[15, 2:].astype(float).tolist(),
        "ev":         B.iloc[16, 2:].astype(float).tolist(),
        "new_fossil": B.iloc[20, 2:].astype(float).tolist(),
        "new_ev":     B.iloc[21, 2:].astype(float).tolist(),
    })
    fleet["total"]         = fleet["fossil"] + fleet["ev"]
    fleet["ev_share_fleet"] = fleet["ev"] / fleet["total"]
    fleet["ev_share_new"]   = fleet["new_ev"] / (fleet["new_ev"] + fleet["new_fossil"])

    return A, drivers, fleet


# ════════════════════════════════════════════════════════════════════════════
#  ELASTICITY ESTIMATION
# ════════════════════════════════════════════════════════════════════════════

def add_elasticities(drivers: pd.DataFrame, simplified: bool = False) -> pd.DataFrame:
    """
    Compute segment-specific price elasticities for every driver and attach
    them as eps_{segment} columns.

    Implements Eq. (24) from the Technical Appendix. The elasticity is a
    linear combination of driver characteristics (income, age, home location)
    and segment attributes (zone, time of day) plus interaction terms.

    Parameters
    ----------
    drivers : pd.DataFrame
        Wide driver table from load_data().
    simplified : bool
        If True, use PARAMS_INCOME_ONLY (no age or home-location terms).
        Useful for robustness comparisons.

    Returns
    -------
    pd.DataFrame
        Input table with four new columns: eps_city-peak, eps_city-offpeak,
        eps_countryside-peak, eps_countryside-offpeak.
    """
    params = PARAMS_INCOME_ONLY if simplified else PARAMS_FULL
    out = drivers.copy()

    for seg in SEGMENTS:
        zone = SEGMENT_ATTRS[seg]["zone"]   # "City" or "Countryside"
        time = SEGMENT_ATTRS[seg]["time"]   # "Peak" or "Off-peak"

        # Start from the baseline elasticity (medium income, middle age,
        # rural resident, off-peak driving)
        eps = params["gamma0"] * np.ones(len(out))

        # Income heterogeneity: lower-income households are more price-
        # sensitive (fewer substitutes, tighter budgets)
        eps += params.get("beta_low",  0) * out["income_mapped"].eq("low").to_numpy()
        eps += params.get("beta_high", 0) * out["income_mapped"].eq("high").to_numpy()

        if not simplified:
            # Age effects
            eps += params.get("beta_young", 0) * out["age_group_mapped"].eq("young").to_numpy()
            eps += params.get("beta_old",   0) * out["age_group_mapped"].eq("old").to_numpy()
            # Urban residents have better access to public transport alternatives
            eps += params.get("beta_urban", 0) * out["home_location"].eq("Home_city").to_numpy()

        # Segment-level adjustments
        eps += params.get("delta_peak", 0) * (1 if time == "Peak" else 0)
        eps += params.get("delta_city", 0) * (1 if zone == "City" else 0)

        # Interaction terms: e.g. low-income off-peak trips may be
        # non-work related and therefore more discretionary (more elastic)
        eps += params.get("beta_low_offpeak", 0) * (out["income_mapped"].eq("low").to_numpy() * (1 if time == "Off-peak" else 0))
        eps += params.get("beta_high_peak",   0) * (out["income_mapped"].eq("high").to_numpy() * (1 if time == "Peak" else 0))
        eps += params.get("delta_city_peak",  0) * (1 if (zone == "City" and time == "Peak") else 0)

        if not simplified:
            # Urban resident who drives in the city: even more alternatives available
            eps += params.get("delta_urban_city", 0) * (out["home_location"].eq("Home_city").to_numpy() * (1 if zone == "City" else 0))

        out[f"eps_{seg}"] = eps

    return out


# ════════════════════════════════════════════════════════════════════════════
#  POLICY SIMULATION
# ════════════════════════════════════════════════════════════════════════════

def run_policy(
    year: int,
    per_km_taxes: dict[str, float] | None = None,
    annual_fee: float = 0.0,
    city_fee: float = 0.0,
    simplified: bool = False,
) -> dict[str, float]:
    """
    Simulate a road-pricing policy and compute national-level welfare effects.

    The simulation runs in four sequential phases that mirror the Technical
    Appendix (Steps 1–21):

      Phase 1 – Behavioural response: new VKT = base × (p1/p0)^ε
                Uses per-km taxes to compute the post-tax price ratio and
                applies the power-law demand response.

      Phase 2 – Per-km dropout gate: if the VKT reduction exceeds 50 %
                (alpha = 0.5) for a given segment, the driver stops driving
                in that segment entirely. This captures the extensive margin
                (e.g., mode switch, relocation).

      Phase 3 – City-fee gate: if the annual city fee exceeds 50 % of the
                driver's remaining city driving cost, they exit city driving.

      Phase 4 – Annual-fee gate: same 50 % threshold applied to total
                remaining driving cost across all segments.

    Welfare decomposition (all scaled to national level via omega weights):
      • Consumer surplus change: area under the demand curve lost due to the tax
      • Government revenue: per-km taxes × post-tax VKT + fixed fees
      • External gains: reduction in congestion + pollution externalities
      • Net welfare = CS change + revenue + external gains

    Parameters
    ----------
    year : int
        Policy year. Determines fleet composition (EV share) and total VKT
        through the omega scaling weights from fleet projections.
    per_km_taxes : dict[str, float] | None
        Per-km tax (DKK) by segment. Missing segments default to 0.
    annual_fee : float
        Annual fixed fee charged to every participating driver (DKK/year).
    city_fee : float
        Annual fixed fee charged only to drivers using city roads (DKK/year).
    simplified : bool
        If True, use the income-only elasticity specification.

    Returns
    -------
    dict with keys:
        year, simplified,
        revenue_bn_dkk, consumer_surplus_bn_dkk, external_gains_bn_dkk,
        welfare_bn_dkk, vkt_reduction_pct, city_peak_reduction_pct,
        dropout_pct_sample
    """
    A, drivers, fleet = load_data()
    df = add_elasticities(drivers, simplified=simplified)

    if per_km_taxes is None:
        per_km_taxes = {s: 0.0 for s in SEGMENTS}

    n_ev   = int((df["car"] == "Electricity").sum())
    n_foss = int((df["car"] == "Fossil").sum())
    p0_arr = df["p0"].to_numpy()   # baseline per-km cost for each driver

    # ── Phase 1 & 2: per-km behavioural response + dropout gate ──────────────
    vkt_star  = {}   # post-tax VKT (continuous response)
    vkt_after1 = {}  # post-tax VKT after extensive-margin dropout
    final_part = {s: np.ones(len(df), dtype=int) for s in SEGMENTS}  # 1 = participating

    for seg in SEGMENTS:
        base = df[seg].to_numpy()
        eps  = df[f"eps_{seg}"].to_numpy()
        t    = per_km_taxes.get(seg, 0.0)
        tau  = (p0_arr + t) / p0_arr            # price ratio p1/p0

        v_star = base * np.power(tau, eps)       # power-law demand response

        # Fraction of VKT reduction: drivers who reduce by more than alpha=0.5
        # are assumed to exit the segment entirely
        red  = np.zeros_like(base, dtype=float)
        nz   = base > 0
        red[nz] = (base[nz] - v_star[nz]) / base[nz]
        keep = (red <= 0.5).astype(int)

        vkt_star[seg]   = v_star
        vkt_after1[seg] = keep * v_star
        final_part[seg] = keep.copy()

    # ── Phase 3: city-fee gate ────────────────────────────────────────────────
    if city_fee > 0:
        city_scope = ["city-peak", "city-offpeak"]
        # Driver's remaining city driving cost after per-km adjustments
        cost_base  = p0_arr * sum(vkt_after1[s] for s in city_scope)
        # Drop out of city driving if city fee > 50 % of residual city cost.
        # np.errstate suppresses the benign divide-by-zero for zero-cost rows;
        # those rows are excluded by the (cost_base > 0) guard anyway.
        with np.errstate(divide="ignore", invalid="ignore"):
            pass_city = ((cost_base > 0) & (city_fee / cost_base <= 0.5)).astype(int)
        for s in city_scope:
            final_part[s] = final_part[s] * pass_city

    # ── Phase 4: annual-fee gate ──────────────────────────────────────────────
    if annual_fee > 0:
        rem_sum   = sum(vkt_after1[s] * final_part[s] for s in SEGMENTS)
        cost_base = p0_arr * rem_sum
        # Drop out entirely if annual fee > 50 % of total residual driving cost
        pass_all  = ((cost_base > 0) & (annual_fee / cost_base <= 0.5)).astype(int)
        for s in SEGMENTS:
            final_part[s] = final_part[s] * pass_all

    # Final VKT: continuous response × participation indicator
    vkt_final = {s: vkt_star[s] * final_part[s] for s in SEGMENTS}

    # Whether each driver participates at all (used for fixed-fee attribution)
    partdrv = np.max(np.column_stack([final_part[s] for s in SEGMENTS]), axis=1)

    # Fixed fees paid (zero for non-participating drivers)
    city_fee_paid   = (np.where(np.max(np.column_stack([final_part[s] for s in ["city-peak", "city-offpeak"]]), axis=1) > 0, city_fee, 0.0)
                       if city_fee else np.zeros(len(df)))
    annual_fee_paid = np.where(partdrv > 0, annual_fee, 0.0) if annual_fee else np.zeros(len(df))
    fees_paid       = city_fee_paid + annual_fee_paid

    # ── Welfare calculations (sample level, then scaled to national) ──────────
    cs      = np.zeros(len(df))
    revenue = np.zeros(len(df))

    for seg in SEGMENTS:
        base = df[seg].to_numpy()
        eps  = df[f"eps_{seg}"].to_numpy()
        t    = per_km_taxes.get(seg, 0.0)
        p1   = p0_arr + t

        # Consumer surplus change: integral under linear-in-log demand curve.
        # For ε ≠ -1: ΔCS = -∫_{p0}^{p1} V(p) dp  (area under demand curve).
        # For ε = -1 (unit elasticity): closed-form log solution.
        cont = np.where(
            np.isclose(eps, -1.0),
            -base * p0_arr * np.log(p1 / p0_arr),
            -base * p0_arr / (eps + 1) * ((p1 / p0_arr) ** (eps + 1) - 1),
        )
        # Drivers who drop out entirely lose their entire consumer surplus
        drop = -p0_arr * base

        cs      += np.where(final_part[seg] == 1, cont, drop)
        revenue += t * vkt_final[seg]

    # Fixed-fee contributions to consumer surplus and revenue
    cs      += -partdrv * fees_paid
    revenue +=  partdrv * fees_paid

    # ── Scale sample → national fleet via omega weights ───────────────────────
    # omega converts one sample driver to the equivalent number of national
    # drivers of that fuel type in the given year.
    fleet_row  = fleet.set_index("year").loc[year]
    omega_ev   = fleet_row["ev"]     / n_ev
    omega_foss = fleet_row["fossil"] / n_foss
    omega      = np.where(df["car"].eq("Electricity"), omega_ev, omega_foss)

    cs_nat  = float((cs  * omega).sum())
    rev_nat = float((revenue * omega).sum())

    # ── National VKT by fuel × segment (for externality calculation) ──────────
    base_nat  = {s: {"Electricity": 0.0, "Fossil": 0.0} for s in SEGMENTS}
    final_nat = {s: {"Electricity": 0.0, "Fossil": 0.0} for s in SEGMENTS}
    for fuel in ["Electricity", "Fossil"]:
        mask = (df["car"] == fuel).to_numpy()
        om   = omega_ev if fuel == "Electricity" else omega_foss
        for seg in SEGMENTS:
            base_nat[seg][fuel]  = float(df.loc[mask, seg].sum() * om)
            final_nat[seg][fuel] = float(vkt_final[seg][mask].sum() * om)

    # ── External gains: congestion + pollution savings ────────────────────────
    # Congestion gains follow a concave functional form (ETA < 1):
    #   ΔEC_congestion = (ext_cost / η) × (ΔV)^η
    # This reflects the non-linear relationship between VKT and congestion
    # (removing the first km from a congested road saves more than the last).
    ec_total = 0.0
    for seg in SEGMENTS:
        area  = "city" if seg.startswith("city") else "countryside"
        V0    = base_nat[seg]["Electricity"]  + base_nat[seg]["Fossil"]
        V1    = final_nat[seg]["Electricity"] + final_nat[seg]["Fossil"]

        # Congestion + accidents + noise + infrastructure savings
        ec_total += OTHER_SUM[seg] / ETA * ((V0 - V1) ** ETA)

        # Pollution savings (fuel-specific)
        ec_total += POLLUTION[("Electricity", area)] * (base_nat[seg]["Electricity"] - final_nat[seg]["Electricity"])
        ec_total += POLLUTION[("Fossil",      area)] * (base_nat[seg]["Fossil"]      - final_nat[seg]["Fossil"])

    # Net social welfare = consumer surplus change + revenue + external gains
    welfare = cs_nat + rev_nat + ec_total

    # ── Summary metrics ───────────────────────────────────────────────────────
    base_total  = sum(base_nat[s]["Electricity"]  + base_nat[s]["Fossil"]  for s in SEGMENTS)
    final_total = sum(final_nat[s]["Electricity"] + final_nat[s]["Fossil"] for s in SEGMENTS)
    city_peak_base  = base_nat["city-peak"]["Electricity"]  + base_nat["city-peak"]["Fossil"]
    city_peak_final = final_nat["city-peak"]["Electricity"] + final_nat["city-peak"]["Fossil"]

    return {
        "year":                     year,
        "simplified":               simplified,
        "revenue_bn_dkk":           rev_nat  / 1e9,
        "consumer_surplus_bn_dkk":  cs_nat   / 1e9,
        "external_gains_bn_dkk":    ec_total / 1e9,
        "welfare_bn_dkk":           welfare  / 1e9,
        "vkt_reduction_pct":        (1 - final_total / base_total) * 100,
        "city_peak_reduction_pct":  (1 - city_peak_final / city_peak_base) * 100,
        "dropout_pct_sample":       float((partdrv == 0).mean() * 100),
    }


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick test: run "Our Proposal" (50% Pigouvian taxes + 1 000 DKK city fee)
    # in 2035 and print the welfare decomposition.
    out = run_policy(
        year=2035,
        per_km_taxes={
            "city-peak":          1.68,   # ~50% of 3.16 DKK/km Pigouvian rate
            "city-offpeak":       0.42,
            "countryside-peak":   0.14,
            "countryside-offpeak": 0.00,
        },
        city_fee=1000,
        simplified=False,
    )
    print(json.dumps(out, indent=2))
