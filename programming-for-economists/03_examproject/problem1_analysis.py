"""
Problem 1: Danish House Prices Analysis
========================================
Analyses the long-run dynamics of Danish house prices using official data
from Danmarks Statistik.  The analysis proceeds in four steps:

    Q1.1 — Nominal price indices by province (EJ56, table API)
            Fetches quarterly one-family house price indices and re-bases
            each series to 100 at 1992Q1 so that cumulative growth is directly
            comparable across regions.

    Q1.2 — Real price indices (CPI-deflated)
            Deflates nominal indices using the monthly CPI (PRIS113), resampled
            to quarterly frequency.  Real prices strip out general inflation
            and reveal genuine changes in housing affordability.

    Q1.3 — Municipality-level data (BM010_houses.xlsx)
            Scatter plot: initial price level (kr/m²) vs total growth.
            A positive correlation would imply that already-expensive areas
            appreciated fastest — a signal of diverging regional housing markets.

    Q1.4 — 4-quarter rolling average and crisis recovery
            Identifies municipalities still trading below their pre-2008 peak.
            The 2008 financial crisis caused a prolonged correction in Danish
            house prices; some peripheral municipalities never fully recovered.

Data sources
------------
    EJ56    : Danmarks Statistik — quarterly one-family house price indices
    PRIS113 : Danmarks Statistik — monthly CPI (all-items, 2015=100)
    BM010_houses.xlsx : Static export of DS table BM010 (kr/m²)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dstapi import DstApi


class DanishHousePrices:
    """Analysis of Danish house prices from Danmarks Statistik."""

    def __init__(self):
        """Initialize storage for data."""
        self.house_prices = None
        self.cpi_data = None
        self.real_prices = None
        self.municipality_data = None

    def fetch_house_prices(self):
        """Fetch house price data from table EJ56 and clean it."""
        api = DstApi('EJ56')
        params = api._define_base_params(language='en')
        df = api.get_data(params=params)

        # Filter for one-family houses and index values
        df = df[(df['EJENDOMSKATE'] == 'One-family houses') &
                (df['TAL'] == 'Index')]

        # API returns strings, convert to numeric
        df['INDHOLD'] = pd.to_numeric(df['INDHOLD'], errors='coerce')

        # Parse quarter strings to datetime
        df['time'] = pd.PeriodIndex(df['TID'], freq='Q').to_timestamp()

        # Remove areas with missing data
        df = df.groupby('OMRÅDE').filter(lambda x: x['INDHOLD'].notna().all())

        # Pivot to wide format
        prices = df.pivot(index='time', columns='OMRÅDE', values='INDHOLD')

        # Index to 100 at 1992Q1
        base = prices.iloc[0]
        prices = (prices / base) * 100

        self.house_prices = prices
        return prices

    def fetch_cpi(self):
        """Get CPI data from table PRIS113."""
        api = DstApi('PRIS113')
        params = api._define_base_params(language='en')
        df = api.get_data(params=params)

        # Convert to numeric
        df['INDHOLD'] = pd.to_numeric(df['INDHOLD'], errors='coerce')

        # Convert TID (e.g., '2020M01') to datetime
        df['time'] = pd.to_datetime(df['TID'], format='%YM%m')
        cpi = df.set_index('time')['INDHOLD']

        self.cpi_data = cpi
        return cpi

    def calculate_real_prices(self):
        """Calculate real prices adjusted for CPI."""
        # CPI is monthly, convert to quarterly by taking mean of available values
        # Use 'ffill' to forward-fill missing months before resampling
        cpi_filled = self.cpi_data.sort_index().asfreq('MS', method='ffill')
        cpi_quarterly = cpi_filled.resample('QE').mean()

        # Align CPI with house prices date range
        cpi_aligned = cpi_quarterly.reindex(self.house_prices.index, method='nearest')

        # Normalize CPI to have 1992Q1 = 100 (to match house price index)
        cpi_index = (cpi_aligned / cpi_aligned.iloc[0]) * 100

        # Real price = nominal price index / CPI index * 100
        # Divide each column by the CPI index (broadcasting along rows)
        real_prices = self.house_prices.div(cpi_index, axis=0) * 100

        self.real_prices = real_prices
        return real_prices

    def rank_provinces(self):
        """Rank provinces by total real price growth."""
        # Growth = last value - first value (using REAL prices)
        growth = self.real_prices.iloc[-1] - self.real_prices.iloc[0]
        return growth.sort_values(ascending=False)

    def plot_nominal_prices(self):
        """Plot nominal house prices over time."""
        fig, ax = plt.subplots(figsize=(12, 6))

        for prov in self.house_prices.columns:
            ax.plot(self.house_prices.index, self.house_prices[prov], label=prov)

        ax.set_title('House Prices by Province (Index: 1992Q1 = 100)')
        ax.set_xlabel('Year')
        ax.set_ylabel('Price Index')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_real_prices(self):
        """Plot real house prices (CPI-adjusted)."""
        fig, ax = plt.subplots(figsize=(12, 6))

        for province in self.real_prices.columns:
            ax.plot(self.real_prices.index, self.real_prices[province], label=province)

        ax.set_title('Real House Prices by Province (CPI-Adjusted)')
        ax.set_xlabel('Year')
        ax.set_ylabel('Real Price Index')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig

    # Q1.3 and Q1.4 methods

    def load_municipality_data(self, filepath='BM010_houses.xlsx'):
        """Load municipality price data from Excel."""
        df = pd.read_excel(filepath, header=None)

        # Row 2 has quarters, row 3+ has municipality data
        quarters = df.iloc[2, 3:].values
        muni_names = df.iloc[3:, 2].values
        price_data = df.iloc[3:, 3:].values

        clean_df = pd.DataFrame(price_data, columns=quarters)
        clean_df.insert(0, 'Municipality', muni_names)

        # Convert prices to numeric
        for q in quarters:
            clean_df[q] = pd.to_numeric(clean_df[q], errors='coerce')

        clean_df = clean_df.dropna()

        self.municipality_data = clean_df
        return clean_df

    def plot_growth_vs_initial(self):
        """Scatter plot: initial price vs total growth."""
        if self.municipality_data is not None:
            df = self.municipality_data
        else:
            df = self.load_municipality_data()

        munis = df['Municipality']
        start_col = df.columns[1]
        end_col = df.columns[-1]

        init_price = df[start_col]
        final_price = df[end_col]
        total_growth = final_price - init_price

        corr = np.corrcoef(init_price, total_growth)[0, 1]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(init_price, total_growth, alpha=0.5)

        # Label top municipalities to avoid clutter
        # Only label: top 5 by growth + top 5 by initial price
        top_growth = total_growth.nlargest(5).index
        top_price = init_price.nlargest(5).index
        label_these = set(top_growth) | set(top_price)

        for idx in label_these:
            ax.text(init_price.iloc[idx], total_growth.iloc[idx],
                    f' {munis.iloc[idx]}', fontsize=8)

        ax.set_xlabel('Initial Price (kr/m²)')
        ax.set_ylabel('Total Growth (kr/m²)')
        ax.set_title(f'Initial Price vs Total Growth (r = {corr:.2f})')
        ax.grid(alpha=0.3)
        plt.tight_layout()

        return fig

    def calculate_rolling_average(self, window=4):
        """4-quarter backward rolling average."""
        if self.municipality_data is not None:
            df = self.municipality_data
        else:
            df = self.load_municipality_data()

        # Get price columns only (skip municipality name)
        price_cols = df.iloc[:, 1:].astype(float)

        # Rolling mean across columns - transpose to make it work
        rolling = price_cols.T.rolling(window=window).mean().T

        # Add municipality names back
        result = pd.DataFrame({'Municipality': df['Municipality']})
        result = pd.concat([result, rolling], axis=1)

        return result

    def analyze_crisis_recovery(self):
        """Find municipalities still below pre-2008 peak."""
        rolling = self.calculate_rolling_average()
        munis = rolling['Municipality']

        qtr_cols = [col for col in rolling.columns if col != 'Municipality']

        # Find 2008Q1 index
        crisis_idx = None
        for i, col in enumerate(qtr_cols):
            if '2008Q1' in str(col):
                crisis_idx = i
                break

        if crisis_idx is None:
            crisis_idx = int(len(qtr_cols) * 0.6)

        pre_crisis = rolling[qtr_cols[:crisis_idx]]
        peaks = pre_crisis.max(axis=1)

        latest = rolling[qtr_cols[-1]]

        still_below = munis[latest < peaks]

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot first 8 municipalities as examples (to keep plot readable)
        for i in range(min(8, len(rolling))):
            ax.plot(rolling.iloc[i, 1:].values, label=munis.iloc[i], alpha=0.7)

        ax.axvline(crisis_idx, color='red', linestyle='--',
                   label='2008 Crisis', alpha=0.5)
        ax.set_xlabel('Quarter Index')
        ax.set_ylabel('Price (kr/m²)')
        ax.set_title('4-Quarter Rolling Average')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        print(f"\nMunicipalities below pre-crisis peak: {len(still_below)}/{len(munis)}")
        if len(still_below) > 0:
            print("\nStill below peak:")
            for m in still_below.values:
                print(f"  - {m}")

        return fig
