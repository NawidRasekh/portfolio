"""
Section 2: International Inflation Comparison
==============================================
Compares Denmark's national CPI (from Danmarks Statistik) against the
Harmonised Index of Consumer Prices (HICP) sourced from FRED/Eurostat,
and places Danish inflation in international context alongside Austria,
the Euro Area, and the United States.

Why two indices?
    CPI (national): Measures price changes for the entire resident population,
    including owner-occupied housing costs. Compiled by Danmarks Statistik.

    HICP (harmonised): Designed for cross-country comparison within the EU.
    Excludes owner-occupied housing so that price levels are comparable across
    member states. Published by Eurostat, accessible via FRED.

    The divergence between CPI and HICP in Denmark is therefore informative:
    when CPI rises faster than HICP, rising housing costs are the likely driver.

Data sources:
    - Danish CPI  : Danmarks Statistik API, table PRIS113
    - HICP (all)  : FRED
      Denmark     CP0000DKM086NEST
      Austria     CP0000ATM086NEST
      Euro Area   CP0000EZ19M086NEST
      USA         CPIAUCSL  (standard CPI, used as external comparator)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader as pdr
from datetime import datetime
from dstapi import DstApi
import warnings

warnings.filterwarnings('ignore', message='API call parameters are not specified')


class InflationAnalysis:
    """
    Full pipeline for cross-country inflation analysis.

    Responsibilities
    ----------------
    1. Fetch Danish CPI from Danmarks Statistik (DstApi)
    2. Fetch HICP data for multiple countries from FRED
    3. Compute 12-month rolling inflation rates (year-on-year % change)
    4. Compare CPI vs HICP for Denmark to assess structural differences
    5. Produce publication-quality visualisations

    Parameters
    ----------
    start_date : str
        ISO date string for the earliest observation included.
        Default '2019-01-01' captures the pre-pandemic baseline as well as
        the 2021-2023 inflationary episode, enabling before/after comparison.
    """

    def __init__(self, start_date='2019-01-01'):

        # Time window: from start_date to today
        self.start_date = start_date
        self.end_date   = datetime.now()

        # FRED series codes for HICP (EU harmonised) or CPI (USA)
        # EU series use base year 2015=100, consistent with Eurostat convention
        self.fred_codes = {
            'Denmark':       'CP0000DKM086NEST',
            'Austria':       'CP0000ATM086NEST',
            'Euro_Area':     'CP0000EZ19M086NEST',
            'United_States': 'CPIAUCSL',
        }

        # Data containers — populated by the fetch/compute methods below
        self.cpi_dst      = None   # Danish CPI from Danmarks Statistik
        self.hicp_data    = None   # HICP panel (months × countries)
        self.comparison_df = None  # Merged CPI/HICP for Denmark
        self.inflation_12m = None  # 12-month % change in HICP

    # ------------------------------------------------------------------
    # 1. DATA FETCHING
    # ------------------------------------------------------------------

    def get_danish_cpi(self):
        """
        Fetch Danish CPI from Danmarks Statistik table PRIS113.

        PRIS113 publishes multiple sub-indices (food, energy, etc.).
        We filter to TYPE == 'INDEKS' to get the all-items headline index —
        the same figure Danmarks Statistik reports to Eurostat as national CPI.

        Returns
        -------
        pd.DataFrame
            Monthly CPI index values indexed by datetime.
        """
        dst = DstApi('PRIS113')
        params = dst._define_base_params(language='en')

        # Restrict to the headline all-items index (2015=100)
        for var in params['variables']:
            if var['code'] == 'TYPE':
                var['values'] = ['INDEKS']

        data = dst.get_data(params=params)

        # TID is formatted '2020M01' — parse to proper datetime
        data['date']    = pd.to_datetime(data['TID'], format='%YM%m')
        data['INDHOLD'] = pd.to_numeric(data['INDHOLD'], errors='coerce')
        data = data.rename(columns={'INDHOLD': 'CPI'})
        data = data.sort_values('date')
        data = data[data['date'] >= self.start_date]

        self.cpi_dst = data[['date', 'CPI']].set_index('date')
        return self.cpi_dst

    def get_hicp_data(self):
        """
        Fetch HICP (and US CPI) from FRED for all countries in self.fred_codes.

        All EU series are monthly, base 2015=100.
        US CPI (CPIAUCSL) uses a different base year but is included for
        directional comparison of the global inflationary cycle.

        Returns
        -------
        pd.DataFrame
            Wide-format panel: rows = months, columns = countries.
        """
        data_dict = {}
        for country, code in self.fred_codes.items():
            df = pdr.get_data_fred(code, start=self.start_date,
                                   end=self.end_date)
            df.columns = [country]
            data_dict[country] = df

        self.hicp_data = pd.concat(data_dict.values(), axis=1)
        return self.hicp_data

    # ------------------------------------------------------------------
    # 2. COMPUTATION
    # ------------------------------------------------------------------

    def compare_cpi_hicp(self):
        """
        Merge Danish CPI (national) with Danish HICP (harmonised) for direct
        comparison on the same time axis.

        The persistent gap (CPI − HICP) primarily reflects owner-occupied
        housing: CPI includes an imputed rent estimate; HICP does not.
        Tracking this difference helps decompose how much of Denmark's
        headline inflation is housing-driven versus broader price pressure.

        Returns
        -------
        pd.DataFrame
            Columns: CPI_DST, HICP_FRED, Difference.
        """
        cpi_filtered = (
            self.cpi_dst[self.cpi_dst.index >= self.start_date]
            .copy()
            .rename(columns={'CPI': 'CPI_DST'})
        )
        hicp_dk = self.hicp_data[['Denmark']].copy()
        hicp_dk.columns = ['HICP_FRED']

        self.comparison_df = pd.merge(
            cpi_filtered, hicp_dk,
            left_index=True, right_index=True, how='inner'
        )
        # Housing wedge: positive value means CPI > HICP
        self.comparison_df['Difference'] = (
            self.comparison_df['CPI_DST'] - self.comparison_df['HICP_FRED']
        )
        return self.comparison_df

    def compute_inflation(self):
        """
        Compute 12-month rolling inflation as the year-on-year % change
        in index levels.

        Using pct_change(periods=12) avoids seasonal distortions that arise
        from month-on-month changes (e.g. energy spikes in winter months).
        fill_method=None ensures missing values propagate rather than being
        silently forward-filled, which would bias the measured inflation rate.

        Returns
        -------
        pd.DataFrame
            12-month inflation rates in percentage points.
        """
        self.inflation_12m = self.hicp_data.pct_change(
            periods=12, fill_method=None
        ) * 100
        return self.inflation_12m

    def get_statistics_by_year(self):
        """
        Summarise inflation by calendar year (min, max, mean).

        Useful for comparing peak inflation across the 2021-2023 episode
        and assessing the cross-country dispersion in price pressures.
        """
        if self.inflation_12m is None:
            self.compute_inflation()

        df = self.inflation_12m.copy()
        df['year'] = df.index.year
        return df.groupby('year').agg(['min', 'max', 'mean'])

    def analyze_comparability(self):
        """
        Quantify how closely the Danish CPI tracks the HICP for Denmark.

        A high correlation (> 0.99) confirms both indices measure fundamentally
        the same phenomenon and are interchangeable for most analyses.
        A persistent mean absolute difference signals that housing costs are
        systematically included in one but not the other.

        Returns
        -------
        dict with keys:
            correlation       – Pearson r between 12m CPI and 12m HICP inflation
            mean_abs_diff     – Average absolute index-point gap
            mean_rel_diff_pct – Gap as % of mean CPI level
        """
        if self.comparison_df is None:
            self.compare_cpi_hicp()

        inflation_cpi  = self.comparison_df['CPI_DST'].pct_change(
            periods=12, fill_method=None) * 100
        inflation_hicp = self.comparison_df['HICP_FRED'].pct_change(
            periods=12, fill_method=None) * 100

        correlation   = inflation_cpi.corr(inflation_hicp)
        self.comparison_df['Difference'] = (
            self.comparison_df['CPI_DST'] - self.comparison_df['HICP_FRED']
        )
        mean_abs_diff = abs(self.comparison_df['Difference']).mean()

        return {
            'correlation':       correlation,
            'mean_abs_diff':     mean_abs_diff,
            'mean_rel_diff_pct': (
                mean_abs_diff / self.comparison_df['CPI_DST'].mean() * 100
            ),
        }

    # ------------------------------------------------------------------
    # 3. VISUALISATION
    # ------------------------------------------------------------------

    def plot_hicp_levels(self, save_path='figures/hicp_levels.png'):
        """
        Plot raw HICP index levels for all countries.

        Visualising levels (rather than rates) shows cumulative price drift
        since the 2015 base year, making it easy to identify which countries
        experienced sustained above-trend inflation during 2021-2023.
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        for col in self.hicp_data.columns:
            ax.plot(self.hicp_data.index, self.hicp_data[col],
                    label=col, linewidth=2)

        ax.set_title('HICP Index Levels: Cross-Country Comparison',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Index (2015=100)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_inflation_rates(self, save_path='figures/inflation_12m.png'):
        """
        Plot 12-month inflation rates for all countries.

        Year-on-year rates are the standard headline measure reported by
        central banks; the ECB's 2% target is defined in terms of euro-area HICP.
        Including Denmark, Austria, and the USA provides context on whether
        Danish inflation was an idiosyncratic or a global phenomenon.
        """
        if self.inflation_12m is None:
            self.compute_inflation()

        fig, ax = plt.subplots(figsize=(14, 6))

        for col in self.inflation_12m.columns:
            ax.plot(self.inflation_12m.index, self.inflation_12m[col],
                    label=col, linewidth=2, alpha=0.8)

        # Zero line separates inflation from deflation
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_title('12-Month Inflation Rate: Cross-Country Comparison',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Inflation Rate (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_cpi_hicp_comparison(self,
                                  save_path='figures/cpi_vs_hicp.png'):
        """
        Two-panel comparison of Danish CPI vs HICP.

        Top panel   : Index levels — shows how the two measures have diverged
                      over time in absolute terms.
        Bottom panel: Difference (CPI − HICP) — isolates the housing-cost wedge
                      that CPI captures but HICP excludes.

        A widening gap implies that owner-occupied housing costs have been
        rising faster than the general price level — a structurally important
        finding for housing-market and monetary-policy analysis.
        """
        if self.comparison_df is None:
            self.compare_cpi_hicp()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Upper panel: both indices on the same scale
        ax1.plot(self.comparison_df.index, self.comparison_df['CPI_DST'],
                 label='CPI (Danmarks Statistik)', linewidth=2, color='blue')
        ax1.plot(self.comparison_df.index, self.comparison_df['HICP_FRED'],
                 label='HICP (FRED/Eurostat)', linewidth=2,
                 color='orange', linestyle='--')
        ax1.set_title('Denmark: CPI vs HICP Index Levels',
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Index (2015=100)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Lower panel: housing wedge (CPI − HICP); positive = housing pushing up CPI
        ax2.plot(self.comparison_df.index, self.comparison_df['Difference'],
                 label='CPI − HICP (housing wedge)', linewidth=2, color='green')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_title('Difference: CPI − HICP (Index Points)',
                      fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Index Point Difference')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
