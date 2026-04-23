# section 2: international inflation comparison

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader as pdr
from datetime import datetime
from dstapi import DstApi
import warnings

# suppress dstapi warnings
warnings.filterwarnings('ignore', message='API call parameters are not specified')

class InflationAnalysis:
    """ inflation analysis across countries """
    
    def __init__(self, start_date='2019-01-01'):
        """ setup """
        
        # a. time period
        self.start_date = start_date
        self.end_date = datetime.now()
        
        # b. fred codes
        self.fred_codes = {
            'Denmark': 'CP0000DKM086NEST',
            'Austria': 'CP0000ATM086NEST',
            'Euro_Area': 'CP0000EZ19M086NEST',
            'United_States': 'CPIAUCSL'
        }
        
        # c. data
        self.cpi_dst = None
        self.hicp_data = None
        self.comparison_df = None
        self.inflation_12m = None
    
    def get_danish_cpi(self):
        """ get danish cpi """
        
        # a. connect to api
        dst = DstApi('PRIS113')
        params = dst._define_base_params(language='en')
        for var in params['variables']:
            if var['code'] == 'TYPE':
                var['values'] = ['INDEKS']
        data = dst.get_data(params=params)
        
        # b. process data
        data['date'] = pd.to_datetime(data['TID'], format='%YM%m')
        data['INDHOLD'] = pd.to_numeric(data['INDHOLD'], errors='coerce')
        data = data.rename(columns={'INDHOLD': 'CPI'})
        data = data.sort_values('date')
        
        # filter by start date
        data = data[data['date'] >= self.start_date]
        
        # c. store
        self.cpi_dst = data[['date', 'CPI']].set_index('date')
        return self.cpi_dst
    
    def get_hicp_data(self):
        """ get hicp from fred """
        
        data_dict = {}
        for country, code in self.fred_codes.items():
            df = pdr.get_data_fred(code, start=self.start_date, end=self.end_date)
            df.columns = [country]
            data_dict[country] = df
        
        self.hicp_data = pd.concat(data_dict.values(), axis=1)
        return self.hicp_data
    
    def compare_cpi_hicp(self):
        """ compare cpi with hicp """
        
        # a. filter cpi
        cpi_filtered = self.cpi_dst[self.cpi_dst.index >= self.start_date].copy()
        cpi_filtered = cpi_filtered.rename(columns={'CPI': 'CPI_DST'})
        
        # b. get hicp for denmark
        hicp_dk = self.hicp_data[['Denmark']].copy()
        hicp_dk.columns = ['HICP_FRED']
        
        # c. merge
        self.comparison_df = pd.merge(cpi_filtered, hicp_dk, 
                                      left_index=True, right_index=True, how='inner')
        
        # d. compute difference
        self.comparison_df['Difference'] = self.comparison_df['CPI_DST'] - self.comparison_df['HICP_FRED']
        
        return self.comparison_df
    
    def compute_inflation(self):
        """ compute 12-month inflation """
        
        self.inflation_12m = self.hicp_data.pct_change(periods=12, fill_method=None) * 100
        return self.inflation_12m
    
    def get_statistics_by_year(self):
        """ statistics by year """
        
        if self.inflation_12m is None:
            self.compute_inflation()
        
        df = self.inflation_12m.copy()
        df['year'] = df.index.year
        return df.groupby('year').agg(['min', 'max', 'mean'])
    
    def analyze_comparability(self):
        """ analyze cpi vs hicp """
        
        if self.comparison_df is None:
            self.compare_cpi_hicp()
        
        # a. calculate inflation from both
        inflation_cpi = self.comparison_df['CPI_DST'].pct_change(periods=12, fill_method=None) * 100
        inflation_hicp = self.comparison_df['HICP_FRED'].pct_change(periods=12, fill_method=None) * 100
        
        # b. correlation
        correlation = inflation_cpi.corr(inflation_hicp)
        
        # c. differences
        self.comparison_df['Difference'] = self.comparison_df['CPI_DST'] - self.comparison_df['HICP_FRED']
        mean_abs_diff = abs(self.comparison_df['Difference']).mean()
        
        return {
            'correlation': correlation,
            'mean_abs_diff': mean_abs_diff,
            'mean_rel_diff_pct': (mean_abs_diff / self.comparison_df['CPI_DST'].mean() * 100)
        }
    
    def plot_hicp_levels(self, save_path='figures/hicp_levels.png'):
        """ plot hicp levels """
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for col in self.hicp_data.columns:
            ax.plot(self.hicp_data.index, self.hicp_data[col], label=col, linewidth=2)
        
        ax.set_title('HICP Index Levels: Cross-Country Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Index')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_inflation_rates(self, save_path='figures/inflation_12m.png'):
        """ plot inflation rates """
        
        if self.inflation_12m is None:
            self.compute_inflation()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for col in self.inflation_12m.columns:
            ax.plot(self.inflation_12m.index, self.inflation_12m[col], 
                   label=col, linewidth=2, alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_title('12-Month Inflation Rate: Cross-Country Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Inflation Rate (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_cpi_hicp_comparison(self, save_path='figures/cpi_vs_hicp.png'):
        """ plot cpi vs hicp """
        
        if self.comparison_df is None:
            self.compare_cpi_hicp()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # a. plot levels
        ax1.plot(self.comparison_df.index, self.comparison_df['CPI_DST'], 
                label='CPI (Danmarks Statistik)', linewidth=2, color='blue')
        ax1.plot(self.comparison_df.index, self.comparison_df['HICP_FRED'], 
                label='HICP (FRED/Eurostat)', linewidth=2, color='orange', linestyle='--')
        ax1.set_title('Denmark: CPI vs HICP Index Levels', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Index (2015=100)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # b. plot difference
        ax2.plot(self.comparison_df.index, self.comparison_df['Difference'], 
                label='CPI - HICP', linewidth=2, color='green')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_title('Difference: CPI - HICP (Index Points)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Index Point Difference')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
