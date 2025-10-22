import pandas as pd
import numpy as np
from datetime import datetime
import os

class DataProcessor:
    """
    Handles data loading, merging, and feature engineering for market sentiment prediction
    """
    
    def __init__(self):
        self.data_dir = 'data'
        
    def load_fii_data(self):
        """Load FII (Foreign Institutional Investment) data"""
        try:
            fii_path = os.path.join(self.data_dir, 'fii_data.csv')
            fii_data = pd.read_csv(fii_path)
            fii_data['date'] = pd.to_datetime(fii_data['date'])
            return fii_data
        except Exception as e:
            raise Exception(f"Error loading FII data: {str(e)}")
    
    def load_oi_data(self):
        """Load Options Open Interest data"""
        try:
            oi_path = os.path.join(self.data_dir, 'oi_data.csv')
            oi_data = pd.read_csv(oi_path)
            oi_data['date'] = pd.to_datetime(oi_data['date'])
            return oi_data
        except Exception as e:
            raise Exception(f"Error loading OI data: {str(e)}")
    
    def merge_data(self, fii_data, oi_data):
        """Merge FII and OI data on date"""
        try:
            merged_data = pd.merge(fii_data, oi_data, on='date', how='inner')
            merged_data = merged_data.sort_values('date')
            return merged_data
        except Exception as e:
            raise Exception(f"Error merging data: {str(e)}")
    
    def engineer_features(self, data):
        """Create engineered features for machine learning"""
        try:
            # Create a copy to avoid modifying original data
            df = data.copy()
            
            # Feature 1: Change in FII net position (ΔFII)
            df['fii_change'] = df['fii_net'].diff()
            
            # Feature 2: Put/Call OI ratio (PCR) - already in data
            # Ensure PCR is numeric
            df['pcr'] = pd.to_numeric(df['pcr'], errors='coerce')
            
            # Feature 3: 3-day moving averages
            df['fii_3d_avg'] = df['fii_net'].rolling(window=3, min_periods=1).mean()
            df['pcr_3d_avg'] = df['pcr'].rolling(window=3, min_periods=1).mean()
            
            # Additional features for better prediction
            df['fii_volatility'] = df['fii_net'].rolling(window=5, min_periods=1).std()
            df['pcr_change'] = df['pcr'].diff()
            df['fii_momentum'] = df['fii_net'] - df['fii_3d_avg']
            
            # Create binary target variable (1 for bullish, 0 for bearish)
            # Logic: Bullish if FII net is positive and PCR is below 1.15 (more calls than puts)
            df['target'] = ((df['fii_net'] > 0) & (df['pcr'] < 1.15)).astype(int)
            
            # Fill NaN values with forward fill, then backward fill for any remaining NaN
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except Exception as e:
            raise Exception(f"Error in feature engineering: {str(e)}")
    
    def load_and_merge_data(self):
        """Complete data loading and processing pipeline"""
        try:
            # Load individual datasets
            fii_data = self.load_fii_data()
            oi_data = self.load_oi_data()
            
            # Merge data
            merged_data = self.merge_data(fii_data, oi_data)
            
            # Engineer features
            processed_data = self.engineer_features(merged_data)
            
            return processed_data
            
        except Exception as e:
            raise Exception(f"Error in data processing pipeline: {str(e)}")
    
    def get_feature_columns(self):
        """Return list of feature columns for model training"""
        return [
            'fii_net',
            'fii_change', 
            'pcr',
            'fii_3d_avg',
            'pcr_3d_avg',
            'fii_volatility',
            'pcr_change',
            'fii_momentum'
        ]
    
    def validate_data(self, data):
        """Validate processed data quality"""
        issues = []
        
        # Check for missing values in key columns
        key_columns = ['date', 'fii_net', 'pcr', 'target']
        for col in key_columns:
            if data[col].isnull().any():
                issues.append(f"Missing values found in {col}")
        
        # Check for duplicate dates
        if data['date'].duplicated().any():
            issues.append("Duplicate dates found")
        
        # Check date range
        date_range = data['date'].max() - data['date'].min()
        if date_range.days < 30:
            issues.append("Data range is less than 30 days")
        
        # Check target distribution
        target_distribution = data['target'].value_counts(normalize=True)
        if len(target_distribution) < 2:
            issues.append("Target variable has only one class")
        elif target_distribution.min() < 0.1:
            issues.append("Severe class imbalance in target variable")
        
        return issues if issues else None
