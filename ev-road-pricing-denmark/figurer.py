"""
figurer.py  –  Standalone figure generator for PCC 2026 analysis
================================================================
Generates ALL figures as individual high-resolution PNGs in  figures/

Run:   python figurer.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "font.size": 11,
    "savefig.dpi": 250,
    "savefig.bbox": "tight",
})

OUTDIR = Path("figures")
OUTDIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  1.  DATA LOADING  (mirrors analysis.ipynb cells 1–3)
# ═══════════════════════════════════════════════════════════════════════════════

XLSX = Path("data/pcc2026/Dataset.xlsx")

# Sheet A
A = pd.read_excel(XLSX, sheet_name="Sheet A", header=13)
A = A[A["id"].notna()].copy()
A["id"] = A["id"].astype(int)
A["distance_yearly"] = pd.to_numeric(A["distance_yearly"], errors="coerce").fillna(0.0)
A["segment"] = (A["zone"].str.lower() + "-" +
                A["time_of_day"].str.lower()
                .str.replace("off-peak", "offpeak")
                .str.replace(r"(?<![f])peak", "peak", regex=True))

# Sheet B
B = pd.read_excel(XLSX, sheet_name="Sheet B", header=None)
years_list = [int(y) for y in B.iloc[14, 2:12].tolist()]
fleet = pd.DataFrame({
    "year":       years_list,
    "fossil":     B.iloc[15, 2:12].astype(float).values,
    "ev":         B.iloc[16, 2:12].astype(float).values,
    "new_fossil": B.iloc[20, 2:12].astype(float).values,
    "new_ev":     B.iloc[21, 2:12].astype(float).values,
})
fleet["total"]     = fleet["fossil"] + fleet["ev"]
fleet["ev_share"]  = fleet["ev"] / fleet["total"] * 100

# Sheet C  –  internal costs (sum of Fuel + Tires + Repairs + Depreciation + Battery)
C = pd.read_excel(XLSX, sheet_name="Sheet C", header=None)
INTERNAL_COSTS = {
    "Fossil":      sum(float(C.iloc[15, c]) for c in range(2, 7)),   # 0.924
    "Electricity": sum(float(C.iloc[16, c]) for c in range(2, 7)),   # 0.870
}
P0 = INTERNAL_COSTS

# Sheet D
D = pd.read_excel(XLSX, sheet_name="Sheet D", header=None)

SEGMENTS = ["city-peak", "city-offpeak", "countryside-peak", "countryside-offpeak"]
SEG_ATTRS = {
    "city-peak":            {"zone": "City",        "time": "Peak",     "area": "city"},
    "city-offpeak":         {"zone": "City",        "time": "Off-peak", "area": "city"},
    "countryside-peak":     {"zone": "Countryside", "time": "Peak",     "area": "countryside"},
    "countryside-offpeak":  {"zone": "Countryside", "time": "Off-peak", "area": "countryside"},
}

# Sheet D layout (inspected):
#   Row 14: headers  Congestion(col3), Accidents(col4), Noise(col5), Infrastructure(col6)
#   Row 15: City/Peak        row 16: City/Off-peak
#   Row 17: Countryside/Peak row 18: Countryside/Off-peak
#   Row 22: Fossil/City pollution=0.15   Row 23: Fossil/Countryside=0.05
#   Row 24: EV/City=0.01                Row 25: EV/Countryside=0.01
#   Row 29: CO2 Fossil = 133.7 g/km     Row 30: CO2 EV = 0

seg_rows = {"city-peak": 15, "city-offpeak": 16, "countryside-peak": 17, "countryside-offpeak": 18}
ext_cols = {"congestion": 3, "accidents": 4, "noise": 5, "infrastructure": 6}

OTHER_EXTERNAL = {}
for seg, row_idx in seg_rows.items():
    OTHER_EXTERNAL[seg] = sum(float(D.iloc[row_idx, ext_cols[k]])
                              for k in ["congestion", "accidents", "noise", "infrastructure"])

POLLUTION = {
    ("Fossil", "city"):         float(D.iloc[22, 3]),
    ("Fossil", "countryside"):  float(D.iloc[23, 3]),
    ("Electricity", "city"):    float(D.iloc[24, 3]),
    ("Electricity", "countryside"): float(D.iloc[25, 3]),
}

CO2 = {"Fossil": float(D.iloc[29, 2]), "Electricity": 0.0}

OTHER_SUM = {seg: OTHER_EXTERNAL[seg] for seg in SEGMENTS}

ETA   = 0.95
ALPHA = 0.50

# ── Driver table ──────────────────────────────────────────────────────────────
seg_nice = {"city-peak": "City – Peak", "city-offpeak": "City – Off-peak",
            "countryside-peak": "Countryside – Peak", "countryside-offpeak": "Countryside – Off-peak"}

seg_labels = {s: f"vkt_{s.replace('-','_')}" for s in SEGMENTS}

# ── Driver table (matches notebook exactly) ───────────────────────────────────
drivers = (A[["id","car","home_location","income_mapped","age_group_mapped"]]
           .drop_duplicates().set_index("id"))
vkt_wide = A.pivot_table(index="id", columns="segment",
                         values="distance_yearly", aggfunc="sum", fill_value=0)
drivers = drivers.join(vkt_wide)
drivers["total_vkt"] = drivers[SEGMENTS].sum(axis=1)
drivers["p0"] = drivers["car"].map(P0)

n_drivers = len(drivers)
n_ev   = (drivers["car"] == "Electricity").sum()
n_foss = (drivers["car"] == "Fossil").sum()

# ── Model parameters (Table 1, Technical Appendix) ───────────────────────────
PARAMS = {
    "gamma0":           -0.40,
    "beta_low":         -0.12,
    "beta_high":         0.06,
    "beta_young":       -0.03,
    "beta_old":          0.03,
    "beta_urban":       -0.03,
    "delta_peak":        0.10,
    "delta_city":       -0.03,
    "delta_city_peak":   0.06,
    "beta_low_offpeak": -0.08,
    "beta_high_peak":    0.04,
    "delta_urban_city": -0.05,
}

PARAMS_SIMPLE = {k: v for k, v in PARAMS.items()
                 if k in ["gamma0","beta_low","beta_high","delta_peak","delta_city",
                          "beta_low_offpeak","beta_high_peak","delta_city_peak"]}

# ═══════════════════════════════════════════════════════════════════════════════
#  2.  MODEL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_elasticities(df, params):
    """Add eps_{seg} columns – Eq. (24) from Technical Appendix."""
    out = df.copy()
    for seg in SEGMENTS:
        zone = SEG_ATTRS[seg]["zone"]   # "City" / "Countryside"
        time = SEG_ATTRS[seg]["time"]   # "Peak" / "Off-peak"

        eps = params["gamma0"] * np.ones(len(out))

        # Income effects
        eps += params.get("beta_low", 0)  * (out["income_mapped"] == "low").to_numpy()
        eps += params.get("beta_high", 0) * (out["income_mapped"] == "high").to_numpy()

        # Age effects
        eps += params.get("beta_young", 0) * (out["age_group_mapped"] == "young").to_numpy()
        eps += params.get("beta_old", 0)   * (out["age_group_mapped"] == "old").to_numpy()

        # Home location
        eps += params.get("beta_urban", 0) * (out["home_location"] == "Home_city").to_numpy()

        # Segment effects
        eps += params.get("delta_peak", 0) * (1 if time == "Peak" else 0)
        eps += params.get("delta_city", 0) * (1 if zone == "City" else 0)

        # Interactions
        eps += params.get("beta_low_offpeak", 0) * ((out["income_mapped"] == "low").to_numpy()  * (1 if time == "Off-peak" else 0))
        eps += params.get("beta_high_peak", 0)   * ((out["income_mapped"] == "high").to_numpy() * (1 if time == "Peak" else 0))
        eps += params.get("delta_city_peak", 0)  * (1 if (zone == "City" and time == "Peak") else 0)
        eps += params.get("delta_urban_city", 0) * ((out["home_location"] == "Home_city").to_numpy() * (1 if zone == "City" else 0))

        out[f"eps_{seg}"] = eps
    return out


def run_policy(df_drivers, year, per_km_taxes=None, annual_fee=0.0, city_fee=0.0,
               params=PARAMS, fleet_df=fleet):
    df = compute_elasticities(df_drivers, params)
    if per_km_taxes is None:
        per_km_taxes = {s: 0.0 for s in SEGMENTS}
    n  = len(df)
    p0 = df["p0"].to_numpy()

    vkt_star  = {}
    vkt_gate1 = {}
    part_seg  = {s: np.ones(n, dtype=int) for s in SEGMENTS}

    for seg in SEGMENTS:
        base = df[seg].to_numpy()
        eps  = df[f"eps_{seg}"].to_numpy()
        t    = per_km_taxes.get(seg, 0.0)
        p1   = p0 + t
        tau  = np.where(p0 > 0, p1 / p0, 1.0)
        v_star = base * np.power(tau, eps)
        r = np.where(base > 0, 1.0 - np.power(tau, eps), 0.0)
        keep = (r <= ALPHA).astype(int)
        vkt_star[seg]  = v_star
        vkt_gate1[seg] = keep * v_star
        part_seg[seg]  = keep

    if city_fee > 0:
        city_segs = ["city-peak", "city-offpeak"]
        C_city = sum(p0 * vkt_gate1[s] for s in city_segs)
        pass_city = np.where(C_city > 0, city_fee / C_city <= ALPHA, False).astype(int)
        for s in city_segs:
            part_seg[s] *= pass_city

    if annual_fee > 0:
        remaining = sum(vkt_gate1[s] * part_seg[s] for s in SEGMENTS)
        C_all = p0 * remaining
        pass_all = np.where(C_all > 0, annual_fee / C_all <= ALPHA, False).astype(int)
        for s in SEGMENTS:
            part_seg[s] *= pass_all

    vkt_final = {s: vkt_star[s] * part_seg[s] for s in SEGMENTS}
    partdrv = np.max(np.column_stack([part_seg[s] for s in SEGMENTS]), axis=1)

    city_fee_paid = np.zeros(n)
    if city_fee > 0:
        part_any_city = np.max(
            np.column_stack([part_seg[s] for s in ["city-peak","city-offpeak"]]), axis=1)
        city_fee_paid = part_any_city * city_fee
    annual_fee_paid = partdrv * annual_fee if annual_fee > 0 else np.zeros(n)
    total_fee = city_fee_paid + annual_fee_paid

    cs = np.zeros(n)
    revenue = np.zeros(n)
    for seg in SEGMENTS:
        base = df[seg].to_numpy()
        eps  = df[f"eps_{seg}"].to_numpy()
        t    = per_km_taxes.get(seg, 0.0)
        p1   = p0 + t
        ratio = np.where(p0 > 0, p1 / p0, 1.0)
        cs_cont = np.where(
            np.isclose(eps, -1.0),
            -base * p0 * np.log(np.maximum(ratio, 1e-12)),
            -base * p0 / (eps + 1) * (np.power(ratio, eps + 1) - 1),
        )
        cs_drop = -p0 * base
        cs += part_seg[seg] * cs_cont + (1 - part_seg[seg]) * cs_drop
        revenue += t * vkt_final[seg]

    cs      -= partdrv * total_fee
    revenue += partdrv * total_fee

    fleet_row  = fleet_df.set_index("year").loc[year]
    omega_ev   = fleet_row["ev"]     / n_ev
    omega_foss = fleet_row["fossil"] / n_foss
    omega = np.where(df["car"] == "Electricity", omega_ev, omega_foss)

    cs_nat  = float((cs * omega).sum())
    rev_nat = float((revenue * omega).sum())

    base_nat  = {s: {} for s in SEGMENTS}
    final_nat = {s: {} for s in SEGMENTS}
    for fuel in ["Electricity", "Fossil"]:
        mask = (df["car"] == fuel).to_numpy()
        om = omega_ev if fuel == "Electricity" else omega_foss
        for seg in SEGMENTS:
            base_nat[seg][fuel]  = float(df.loc[mask, seg].sum() * om)
            final_nat[seg][fuel] = float(vkt_final[seg][mask].sum() * om)

    ec_total = 0.0
    for seg in SEGMENTS:
        area = SEG_ATTRS[seg]["area"]
        V0 = base_nat[seg]["Electricity"] + base_nat[seg]["Fossil"]
        V1 = final_nat[seg]["Electricity"] + final_nat[seg]["Fossil"]
        dV = V0 - V1
        if dV > 0:
            ec_total += OTHER_SUM[seg] / ETA * (dV ** ETA)
        for fuel in ["Electricity", "Fossil"]:
            ec_total += POLLUTION[(fuel, area)] * (base_nat[seg][fuel] - final_nat[seg][fuel])

    welfare = cs_nat + rev_nat + ec_total

    base_total  = sum(base_nat[s]["Electricity"]  + base_nat[s]["Fossil"]  for s in SEGMENTS)
    final_total = sum(final_nat[s]["Electricity"] + final_nat[s]["Fossil"] for s in SEGMENTS)
    cp_base  = base_nat["city-peak"]["Electricity"]  + base_nat["city-peak"]["Fossil"]
    cp_final = final_nat["city-peak"]["Electricity"] + final_nat["city-peak"]["Fossil"]
    co2_base  = sum(base_nat[s]["Fossil"] * CO2["Fossil"] for s in SEGMENTS)
    co2_final = sum(final_nat[s]["Fossil"] * CO2["Fossil"] for s in SEGMENTS)

    return {
        "year": year,
        "revenue_bn": rev_nat / 1e9,
        "cs_bn": cs_nat / 1e9,
        "ec_bn": ec_total / 1e9,
        "welfare_bn": welfare / 1e9,
        "vkt_red_pct": (1 - final_total / base_total) * 100 if base_total > 0 else 0,
        "cp_red_pct":  (1 - cp_final / cp_base) * 100 if cp_base > 0 else 0,
        "dropout_pct": float((partdrv == 0).mean()) * 100,
        "co2_kt": (co2_base - co2_final) / 1e9,
    }


def compute_equity(df_drivers, taxes, annual_fee=0.0, city_fee=0.0, params=PARAMS):
    df = compute_elasticities(df_drivers, params)
    n = len(df)
    p0 = df["p0"].to_numpy()
    part = {s: np.ones(n, dtype=int) for s in SEGMENTS}
    vkt_g1 = {}
    for seg in SEGMENTS:
        base = df[seg].to_numpy()
        eps  = df[f"eps_{seg}"].to_numpy()
        t    = taxes.get(seg, 0.0)
        tau  = np.where(p0 > 0, (p0 + t) / p0, 1.0)
        vs   = base * np.power(tau, eps)
        r    = np.where(base > 0, 1.0 - np.power(tau, eps), 0.0)
        keep = (r <= ALPHA).astype(int)
        vkt_g1[seg] = keep * vs
        part[seg]   = keep
    if city_fee > 0:
        C_city = sum(p0 * vkt_g1[s] for s in ["city-peak", "city-offpeak"])
        pc = np.where(C_city > 0, city_fee / C_city <= ALPHA, False).astype(int)
        for s in ["city-peak", "city-offpeak"]:
            part[s] *= pc
    if annual_fee > 0:
        remaining = sum(vkt_g1[s] * part[s] for s in SEGMENTS)
        C_all = p0 * remaining
        pa = np.where(C_all > 0, annual_fee / C_all <= ALPHA, False).astype(int)
        for s in SEGMENTS:
            part[s] *= pa
    vkt_final = {s: vkt_g1[s] * part[s] for s in SEGMENTS}
    partdrv   = np.max(np.column_stack([part[s] for s in SEGMENTS]), axis=1)
    total_paid = np.zeros(n)
    for seg in SEGMENTS:
        total_paid += taxes.get(seg, 0.0) * vkt_final[seg]
    if city_fee > 0:
        pac = np.max(np.column_stack([part[s] for s in ["city-peak","city-offpeak"]]), axis=1)
        total_paid += pac * city_fee
    if annual_fee > 0:
        total_paid += partdrv * annual_fee
    burden_pct = np.where(
        p0 * df["total_vkt"].values > 0,
        total_paid / (p0 * df["total_vkt"].values) * 100, 0)
    df["burden_pct"] = burden_pct
    grp = df.groupby("income_mapped")["burden_pct"].mean()
    if grp.max() == 0:
        return 1.0
    return float(grp.min() / grp.max())


# ═══════════════════════════════════════════════════════════════════════════════
#  3.  SCENARIOS  &  SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

pigouvian_taxes = {
    "city-peak": 3.31, "city-offpeak": 1.20,
    "countryside-peak": 0.56, "countryside-offpeak": 0.26,
}
our_proposal_taxes = {s: round(v * 0.50, 2) for s, v in pigouvian_taxes.items()}

scenarios = {
    "1. Flat 0.30 DKK/km":          {"taxes": {s: 0.30 for s in SEGMENTS}},
    "2. Flat 0.50 DKK/km":          {"taxes": {s: 0.50 for s in SEGMENTS}},
    "3. Pigouvian (full)":           {"taxes": pigouvian_taxes},
    "4. Pigouvian (50%)":            {"taxes": our_proposal_taxes},
    "5. Congestion focus (city-pk)": {"taxes": {"city-peak": 2.50, "city-offpeak": 0.30,
                                                "countryside-peak": 0.0, "countryside-offpeak": 0.0}},
    "6. Annual fee 5,000":           {"taxes": {s: 0.0 for s in SEGMENTS}, "annual_fee": 5000},
    "7. City fee 2,000":             {"taxes": {s: 0.0 for s in SEGMENTS}, "city_fee": 2000},
    "8. Our Proposal":               {"taxes": our_proposal_taxes, "city_fee": 1000},
}

print("Running simulations …")
results = []
for name, cfg in scenarios.items():
    for par, mlabel in [(PARAMS, "Full"), (PARAMS_SIMPLE, "Simple")]:
        for year in [2026, 2030, 2035]:
            out = run_policy(drivers, year,
                             per_km_taxes=cfg["taxes"],
                             annual_fee=cfg.get("annual_fee", 0.0),
                             city_fee=cfg.get("city_fee", 0.0),
                             params=par)
            out["Scenario"] = name
            out["Model"] = mlabel
            results.append(out)

res_df = pd.DataFrame(results)
sub = res_df[(res_df["year"] == 2035) & (res_df["Model"] == "Full")].copy().set_index("Scenario")
print(f"  ✓ {len(results)} simulation runs complete.\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  4.  PENTAGON DATA
# ═══════════════════════════════════════════════════════════════════════════════

print("Computing pentagon scores …")
pentagon_data = {}
for name, cfg in scenarios.items():
    taxes = cfg["taxes"]
    af    = cfg.get("annual_fee", 0.0)
    cf    = cfg.get("city_fee", 0.0)
    row   = sub.loc[name]
    pentagon_data[name] = {
        "Green\nTransition":        row["co2_kt"],
        "Government\nRevenue":      row["revenue_bn"],
        "Distributional\nEffects":  compute_equity(drivers, taxes, annual_fee=af, city_fee=cf),
        "Economic\nWelfare":        row["welfare_bn"],
        "Mobility":                 100.0 - row["vkt_red_pct"] - row["dropout_pct"] * 2,
    }

pentagon_data["0. Baseline (no tax)"] = {
    "Green\nTransition": 0.0,
    "Government\nRevenue": 0.0,
    "Distributional\nEffects": 1.0,
    "Economic\nWelfare": 0.0,
    "Mobility": 100.0,
}

dimensions = list(next(iter(pentagon_data.values())).keys())
raw = {dim: [pentagon_data[sc][dim] for sc in pentagon_data] for dim in dimensions}
norm_ranges = {}
for dim in dimensions:
    vals = raw[dim]
    lo, hi = min(vals), max(vals)
    norm_ranges[dim] = (lo, hi) if hi != lo else (lo, lo + 1)

def normalise(val, dim):
    lo, hi = norm_ranges[dim]
    return max(0.0, min(10.0, (val - lo) / (hi - lo) * 10.0))

pentagon_norm = {}
for sc in pentagon_data:
    pentagon_norm[sc] = [normalise(pentagon_data[sc][d], d) for d in dimensions]

all_scenarios = ["0. Baseline (no tax)"] + list(scenarios.keys())

# Angles for the 5-vertex polygon
N_DIM = len(dimensions)
angles = np.linspace(0, 2 * np.pi, N_DIM, endpoint=False).tolist()
angles += angles[:1]

# Colour palette
colors_pent = {
    "0. Baseline (no tax)":            "#999999",
    "1. Flat 0.30 DKK/km":            "#1f77b4",
    "2. Flat 0.50 DKK/km":            "#aec7e8",
    "3. Pigouvian (full)":             "#ff7f0e",
    "4. Pigouvian (50%)":              "#ffbb78",
    "5. Congestion focus (city-pk)":   "#2ca02c",
    "6. Annual fee 5,000":             "#d62728",
    "7. City fee 2,000":               "#9467bd",
    "8. Our Proposal":                 "#e377c2",
}

def make_radar(ax, values, label, color, fill_alpha=0.25, lw=2.5):
    vals = values + values[:1]
    ax.plot(angles, vals, color=color, linewidth=lw, label=label)
    ax.fill(angles, vals, color=color, alpha=fill_alpha)

print("  ✓ Pentagon scores ready.\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  5.  FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

saved = []

# ─────────────────────────────────────────────────────────────────────────────
#  5a.  Individual pentagram per scenario  (separate PNG each)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating individual pentagrams …")

for sc in all_scenarios:
    vals = pentagon_norm[sc]
    col  = colors_pent.get(sc, "#333333")
    clean = sc.replace("\n", " ")
    fname = f"pentagram_{clean.split('.')[0].strip()}.png".lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    make_radar(ax, vals, clean, col, fill_alpha=0.30, lw=3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=8, color="grey")
    ax.set_title(clean, fontsize=15, fontweight="bold", pad=25)
    ax.grid(True, alpha=0.3)

    # Value annotations
    for i, (angle, val) in enumerate(zip(angles[:-1], vals)):
        ax.text(angle, val + 0.7, f"{val:.1f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=col)

    path = OUTDIR / fname
    fig.savefig(path)
    plt.close(fig)
    saved.append(path.name)
    print(f"  ✓ {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  5b.  3×3 grid of all pentagrams
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating 3×3 pentagram grid …")

fig, axes = plt.subplots(3, 3, figsize=(18, 18), subplot_kw=dict(polar=True))
fig.suptitle("Policy Pentagrams – Expert Group's Five Considerations (2035)",
             fontsize=18, fontweight="bold", y=0.98)

for idx, sc in enumerate(all_scenarios):
    ax = axes[idx // 3][idx % 3]
    vals = pentagon_norm[sc]
    col  = colors_pent.get(sc, "#333333")
    clean = sc.replace("\n", " ")
    make_radar(ax, vals, clean, col, fill_alpha=0.30, lw=2.5)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=9, fontweight="bold")
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=7, color="grey")
    ax.set_title(clean, fontsize=11, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    for i, (angle, val) in enumerate(zip(angles[:-1], vals)):
        ax.text(angle, val + 0.6, f"{val:.1f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=col)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUTDIR / "pentagrams_individual.png")
plt.close(fig)
saved.append("pentagrams_individual.png")
print("  ✓ pentagrams_individual.png")


# ─────────────────────────────────────────────────────────────────────────────
#  5c.  Overlay: all scenarios on one pentagram
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating overlay pentagram …")

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
fig.suptitle("All Scenarios – Pentagram Comparison (2035)",
             fontsize=16, fontweight="bold", y=1.02)
for sc in all_scenarios:
    col  = colors_pent.get(sc, "#333333")
    clean = sc.replace("\n", " ")
    lw    = 3.5 if sc in ["8. Our Proposal", "0. Baseline (no tax)"] else 1.8
    fa    = 0.20 if sc in ["8. Our Proposal", "0. Baseline (no tax)"] else 0.08
    make_radar(ax, pentagon_norm[sc], clean, col, fill_alpha=fa, lw=lw)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(dimensions, fontsize=11, fontweight="bold")
ax.set_ylim(0, 10)
ax.set_yticks([2, 4, 6, 8, 10])
ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=8, color="grey")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=9, framealpha=0.9)
plt.tight_layout()
fig.savefig(OUTDIR / "pentagrams_overlay.png")
plt.close(fig)
saved.append("pentagrams_overlay.png")
print("  ✓ pentagrams_overlay.png")


# ─────────────────────────────────────────────────────────────────────────────
#  5d.  Welfare decomposition bar chart
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating welfare decomposition …")

plot_df = sub.copy()
fig, ax = plt.subplots(figsize=(12, 6))
y = np.arange(len(plot_df))
h = 0.25
ax.barh(y + h,   plot_df["cs_bn"],       height=h, label="Consumer surplus", color="#d62728", alpha=0.85)
ax.barh(y,       plot_df["revenue_bn"],   height=h, label="Tax revenue",     color="#2ca02c", alpha=0.85)
ax.barh(y - h,   plot_df["ec_bn"],        height=h, label="External savings", color="#1f77b4", alpha=0.85)
ax.scatter(plot_df["welfare_bn"], y, color="black", zorder=5, s=80, marker="D", label="Net welfare ΔW")
ax.set_yticks(y)
ax.set_yticklabels([s.replace("\n", " ") for s in plot_df.index], fontsize=10)
ax.set_xlabel("Billion DKK / year", fontsize=12)
ax.set_title("Welfare Decomposition – All Scenarios (2035, Full Model)", fontsize=14, fontweight="bold")
ax.axvline(0, color="black", lw=0.8)
ax.legend(fontsize=10)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fig.savefig(OUTDIR / "welfare_decomposition_2035.png")
plt.close(fig)
saved.append("welfare_decomposition_2035.png")
print("  ✓ welfare_decomposition_2035.png")


# ─────────────────────────────────────────────────────────────────────────────
#  5e.  Scenario tradeoff scatter
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating tradeoff scatter …")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
pairs = [
    ("revenue_bn", "welfare_bn", "Revenue (bn)", "Welfare ΔW (bn)"),
    ("vkt_red_pct", "welfare_bn", "VKT reduction %", "Welfare ΔW (bn)"),
    ("co2_kt", "revenue_bn", "CO₂ saved (kt)", "Revenue (bn)"),
]
for ax, (xc, yc, xl, yl) in zip(axes, pairs):
    for sc_name in sub.index:
        col = colors_pent.get(sc_name, "#333333")
        ax.scatter(sub.loc[sc_name, xc], sub.loc[sc_name, yc],
                   color=col, s=120, zorder=5, edgecolors="white", linewidths=0.8)
        ax.annotate(sc_name.split(".")[0].strip(),
                    (sub.loc[sc_name, xc], sub.loc[sc_name, yc]),
                    fontsize=8, ha="left", va="bottom",
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.grid(True, alpha=0.3)

fig.suptitle("Scenario Trade-offs (2035, Full Model)", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUTDIR / "scenario_tradeoffs_2035.png")
plt.close(fig)
saved.append("scenario_tradeoffs_2035.png")
print("  ✓ scenario_tradeoffs_2035.png")


# ─────────────────────────────────────────────────────────────────────────────
#  5f.  Fleet projection + welfare timeseries
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating fleet & timeseries …")

ours_time = res_df[(res_df["Scenario"] == "8. Our Proposal") & (res_df["Model"] == "Full")].copy()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# Fleet
ax = axes[0]
ax.stackplot(fleet["year"], fleet["fossil"]/1e6, fleet["ev"]/1e6,
             labels=["Fossil", "EV"], colors=["#d62728", "#2ca02c"], alpha=0.75)
ax.set_ylabel("Vehicles (millions)")
ax.set_title("Fleet Composition")
ax.legend(loc="center right")
ax.grid(True, alpha=0.3)

# Welfare over time
ax = axes[1]
ax.plot(ours_time["year"], ours_time["welfare_bn"], "o-", color="#e377c2", lw=2.5)
ax.set_ylabel("ΔW (bn DKK/yr)")
ax.set_title("'Our Proposal' – Welfare over Time")
ax.grid(True, alpha=0.3)

# Revenue over time
ax = axes[2]
ax.plot(ours_time["year"], ours_time["revenue_bn"], "s-", color="#2ca02c", lw=2.5)
ax.set_ylabel("Revenue (bn DKK/yr)")
ax.set_title("'Our Proposal' – Revenue over Time")
ax.grid(True, alpha=0.3)

fig.suptitle("Fleet Projection & 'Our Proposal' Time Series", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUTDIR / "fleet_welfare_timeseries.png")
plt.close(fig)
saved.append("fleet_welfare_timeseries.png")
print("  ✓ fleet_welfare_timeseries.png")


# ─────────────────────────────────────────────────────────────────────────────
#  5g.  Distribution analysis  – "Our Proposal"
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating distribution figure …")

df_d = compute_elasticities(drivers, PARAMS)
taxes_ours = our_proposal_taxes
city_fee_ours = 1000

p0d = df_d["p0"].to_numpy()
part_d = {s: np.ones(len(df_d), dtype=int) for s in SEGMENTS}
vkt_g1_d = {}
for seg in SEGMENTS:
    base = df_d[seg].to_numpy()
    eps  = df_d[f"eps_{seg}"].to_numpy()
    t    = taxes_ours.get(seg, 0.0)
    tau  = np.where(p0d > 0, (p0d + t) / p0d, 1.0)
    vs   = base * np.power(tau, eps)
    r    = np.where(base > 0, 1.0 - np.power(tau, eps), 0.0)
    keep = (r <= ALPHA).astype(int)
    vkt_g1_d[seg] = keep * vs
    part_d[seg]   = keep

if city_fee_ours > 0:
    C_city = sum(p0d * vkt_g1_d[s] for s in ["city-peak", "city-offpeak"])
    pc = np.where(C_city > 0, city_fee_ours / C_city <= ALPHA, False).astype(int)
    for s in ["city-peak", "city-offpeak"]:
        part_d[s] *= pc

vkt_final_d = {s: vkt_g1_d[s] * part_d[s] for s in SEGMENTS}
partdrv_d = np.max(np.column_stack([part_d[s] for s in SEGMENTS]), axis=1)

df_d["tax_paid"] = sum(taxes_ours.get(seg, 0.0) * vkt_final_d[seg] for seg in SEGMENTS)
part_any_city = np.max(np.column_stack([part_d[s] for s in ["city-peak","city-offpeak"]]), axis=1)
df_d["fee_paid"]   = part_any_city * city_fee_ours
df_d["total_paid"] = df_d["tax_paid"] + df_d["fee_paid"]
df_d["vkt_after"]  = sum(vkt_final_d[s] for s in SEGMENTS)
df_d["vkt_lost"]   = df_d["total_vkt"] - df_d["vkt_after"]
df_d["burden_pct"] = np.where(
    p0d * df_d["total_vkt"].values > 0,
    df_d["total_paid"].values / (p0d * df_d["total_vkt"].values) * 100, 0)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
colors_inc = {"low": "#ef5350", "medium": "#42a5f5", "high": "#66bb6a"}

# Box 1: Tax burden
ax = axes[0]
data = [df_d.loc[df_d["income_mapped"] == g, "total_paid"].values for g in ["low","medium","high"]]
bp = ax.boxplot(data, tick_labels=["Low","Medium","High"], patch_artist=True, showfliers=True, flierprops=dict(marker="o", markersize=4))
for patch, g in zip(bp["boxes"], ["low","medium","high"]):
    patch.set_facecolor(colors_inc[g])
    patch.set_alpha(0.6)
ax.set_ylabel("Total paid (DKK/year)")
ax.set_xlabel("Income group")
ax.set_title("Tax Burden by Income")

# Box 2: VKT lost
ax = axes[1]
data2 = [df_d.loc[df_d["income_mapped"] == g, "vkt_lost"].values for g in ["low","medium","high"]]
bp2 = ax.boxplot(data2, tick_labels=["Low","Medium","High"], patch_artist=True, showfliers=True, flierprops=dict(marker="o", markersize=4))
for patch, g in zip(bp2["boxes"], ["low","medium","high"]):
    patch.set_facecolor(colors_inc[g])
    patch.set_alpha(0.6)
ax.set_ylabel("VKT reduced (km/year)")
ax.set_xlabel("Income group")
ax.set_title("Behavioural Response by Income")

# Histogram: burden %
ax = axes[2]
for g in ["low","medium","high"]:
    vals = df_d.loc[df_d["income_mapped"] == g, "burden_pct"].values
    ax.hist(vals, bins=30, alpha=0.5, color=colors_inc[g], label=g.title(), density=True)
ax.set_xlabel("Burden (% of baseline driving cost)")
ax.set_ylabel("Density")
ax.set_title("Relative Burden Distribution")
ax.legend()

fig.suptitle("Distributional Effects: 'Our Proposal'", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUTDIR / "distribution_our_proposal.png")
plt.close(fig)
saved.append("distribution_our_proposal.png")
print("  ✓ distribution_our_proposal.png")


# ─────────────────────────────────────────────────────────────────────────────
#  5h.  Elasticity distribution figure
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating elasticity figure …")

drivers_full = compute_elasticities(drivers, PARAMS)
eps_cols = [f"eps_{s}" for s in SEGMENTS]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, s in zip(axes.flat, SEGMENTS):
    col = f"eps_{s}"
    vals = drivers_full[col].values
    ax.hist(vals, bins=40, color="#1f77b4", alpha=0.7, edgecolor="white")
    ax.axvline(vals.mean(), color="red", lw=2, ls="--", label=f"Mean = {vals.mean():.3f}")
    ax.set_title(seg_nice[s], fontsize=12, fontweight="bold")
    ax.set_xlabel("Elasticity (ε)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle("Price Elasticity Distributions by Segment (Full Model, Unweighted)",
             fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUTDIR / "elasticities_unweighted.png")
plt.close(fig)
saved.append("elasticities_unweighted.png")
print("  ✓ elasticities_unweighted.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  DONE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 60)
print(f"  ALL DONE – {len(saved)} figures saved to {OUTDIR}/")
print("═" * 60)
for f in sorted(saved):
    print(f"   • {f}")
print()
