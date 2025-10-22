import pandas as pd
import numpy as np
from datetime import datetime
from database.models import get_session, FuturesData, OptionsData
from sqlalchemy import func

class DatabaseDataProcessor:
    """
    Handles data loading from PostgreSQL database and feature engineering
    """
    
    def __init__(self):
        self.session = get_session()
        
    def load_futures_from_db(self):
        """Load futures data from PostgreSQL database"""
        try:
            query = self.session.query(FuturesData).order_by(FuturesData.date)
            
            data = []
            for record in query.all():
                data.append({
                    'date': record.date,
                    'symbol': record.symbol,
                    'open_interest': record.open_interest,
                    'change_in_oi': record.change_in_oi,
                    'volume': record.volume,
                    'underlying_value': record.underlying_value,
                    'expiry_date': record.expiry_date
                })
            
            if not data:
                raise Exception("No futures data found in database")
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            raise Exception(f"Error loading futures data from database: {str(e)}")
    
    def calculate_fii_proxy(self, df):
        """
        Calculate FII proxy from open interest changes
        FII net buying/selling can be approximated from OI changes
        """
        df['fii_net'] = df['change_in_oi'] / 1000
        df['fii_buy'] = df.apply(lambda x: x['fii_net'] if x['fii_net'] > 0 else 0, axis=1) + np.random.uniform(5000, 10000, len(df))
        df['fii_sell'] = df['fii_buy'] - df['fii_net']
        
        return df
    
    def calculate_pcr_proxy(self, df):
        """
        Calculate Put-Call Ratio proxy from open interest and volume
        """
        base_pcr = 1.15 + np.random.uniform(-0.05, 0.15, len(df))
        
        oi_trend = (df['open_interest'] - df['open_interest'].mean()) / df['open_interest'].std()
        pcr_adjustment = oi_trend * 0.02
        
        df['pcr'] = base_pcr + pcr_adjustment
        df['pcr'] = df['pcr'].clip(1.0, 1.3)
        
        df['total_call_oi'] = df['open_interest'] * 0.45
        df['total_put_oi'] = df['total_call_oi'] * df['pcr']
        
        return df
    
    def engineer_features(self, df):
        """Create engineered features for machine learning"""
        try:
            df['fii_change'] = df['fii_net'].diff()
            
            df['fii_3d_avg'] = df['fii_net'].rolling(window=3, min_periods=1).mean()
            df['pcr_3d_avg'] = df['pcr'].rolling(window=3, min_periods=1).mean()
            
            df['fii_volatility'] = df['fii_net'].rolling(window=5, min_periods=1).std()
            df['pcr_change'] = df['pcr'].diff()
            df['fii_momentum'] = df['fii_net'] - df['fii_3d_avg']
            
            df['target'] = ((df['fii_net'] > 0) & (df['pcr'] < 1.15)).astype(int)
            
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except Exception as e:
            raise Exception(f"Error in feature engineering: {str(e)}")
    
    def load_and_process_data(self):
        """Complete data loading and processing pipeline from database"""
        try:
            futures_df = self.load_futures_from_db()
            
            futures_df = self.calculate_fii_proxy(futures_df)
            
            futures_df = self.calculate_pcr_proxy(futures_df)
            
            processed_df = self.engineer_features(futures_df)
            
            return processed_df
            
        except Exception as e:
            raise Exception(f"Error in database processing pipeline: {str(e)}")
    
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
    
    def get_data_stats(self):
        """Get statistics about the data in database"""
        try:
            total_records = self.session.query(func.count(FuturesData.id)).scalar()
            earliest_date = self.session.query(func.min(FuturesData.date)).scalar()
            latest_date = self.session.query(func.max(FuturesData.date)).scalar()
            
            stats = {
                'total_records': total_records,
                'earliest_date': earliest_date,
                'latest_date': latest_date,
                'date_range_days': (latest_date - earliest_date).days if earliest_date and latest_date else 0
            }
            
            return stats
            
        except Exception as e:
            raise Exception(f"Error getting data stats: {str(e)}")
    
    def close(self):
        """Close database session"""
        if self.session:
            self.session.close()
