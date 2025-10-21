import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataProcessor:
    """Data processing utilities for energy consumption analysis"""
    
    def __init__(self):
        self.data = None
        
    def load_data(self, file_path):
        """Load data from CSV file with column filtering"""
        try:
            # Read the full CSV file first
            full_data = pd.read_csv(file_path)
            
            # Define the exact columns to keep from the steel industry dataset
            required_columns = [
                'date',
                'Usage_kWh', 
                'Lagging_Current_Reactive.Power_kVarh',
                'Leading_Current_Reactive.Power_kVarh',
                'CO2(CO2)',
                'Lagging_Current_Power_Factor',
                'Leading_Current_Power_Factor',
                'NSM',
                'WeekStatus',
                'Day_of_week',
                'Load_Type'
            ]
            
            # Keep only the required columns that exist in the dataset
            existing_columns = [col for col in required_columns if col in full_data.columns]
            
            if len(existing_columns) < 3:  # Need at least basic columns
                print(f"Warning: Only found {len(existing_columns)} required columns out of {len(required_columns)}")
                print(f"Available columns: {list(full_data.columns)}")
                print(f"Required columns: {required_columns}")
                # Fallback: keep all columns if we don't find enough required ones
                self.data = full_data
            else:
                # Filter to keep only required columns
                self.data = full_data[existing_columns].copy()
                print(f"✅ Filtered to keep {len(existing_columns)} columns: {existing_columns}")
            
            # Handle different date formats with dayfirst=True for DD/MM/YYYY format
            if 'date' in self.data.columns:
                try:
                    # Try with dayfirst=True for DD/MM/YYYY HH:MM format
                    self.data['Date'] = pd.to_datetime(self.data['date'], dayfirst=True)
                except:
                    try:
                        # Try with explicit format for DD/MM/YYYY HH:MM
                        self.data['Date'] = pd.to_datetime(self.data['date'], format='%d/%m/%Y %H:%M')
                    except:
                        # Fallback to mixed format inference
                        self.data['Date'] = pd.to_datetime(self.data['date'], format='mixed', dayfirst=True)
                
                # Remove original date column
                self.data = self.data.drop(columns=['date'])
                
            elif 'Date' in self.data.columns:
                try:
                    self.data['Date'] = pd.to_datetime(self.data['Date'], dayfirst=True)
                except:
                    try:
                        self.data['Date'] = pd.to_datetime(self.data['Date'], format='%d/%m/%Y %H:%M')
                    except:
                        self.data['Date'] = pd.to_datetime(self.data['Date'], format='mixed', dayfirst=True)
            
            # Remove any remaining original columns that weren't renamed
            columns_to_remove = []
            for col in self.data.columns:
                if col in ['date', 'CO2(CO2)', 'Lagging_Current_Reactive.Power_kVarh', 
                          'Leading_Current_Reactive.Power_kVarh', 'Lagging_Current_Power_Factor',
                          'Leading_Current_Power_Factor', 'WeekStatus', 'Day_of_week'] and col != 'Date':
                    # Only remove if the renamed version exists
                    renamed_version = {
                        'CO2(CO2)': 'CO2_tCO2',
                        'Lagging_Current_Reactive.Power_kVarh': 'Reactive_Power_kVarh',
                        'Leading_Current_Reactive.Power_kVarh': 'Leading_Reactive_Power_kVarh',
                        'Lagging_Current_Power_Factor': 'Power_Factor',
                        'Leading_Current_Power_Factor': 'Leading_Power_Factor',
                        'WeekStatus': 'Week_Status',
                        'Day_of_week': 'Day_of_Week'
                    }.get(col)
                    
                    if renamed_version and renamed_version in self.data.columns:
                        columns_to_remove.append(col)
            
            if columns_to_remove:
                self.data = self.data.drop(columns=columns_to_remove)
                print(f"Removed duplicate original columns: {columns_to_remove}")
            
            # Standardize column names for steel industry data
            column_mapping = {
                'Lagging_Current_Reactive.Power_kVarh': 'Reactive_Power_kVarh',
                'Lagging_Current_Reactive_Power_kVarh': 'Reactive_Power_kVarh',
                'Leading_Current_Reactive.Power_kVarh': 'Leading_Reactive_Power_kVarh', 
                'Leading_Current_Reactive_Power_kVarh': 'Leading_Reactive_Power_kVarh',
                'CO2(CO2)': 'CO2_tCO2',
                'CO2(tCO2)': 'CO2_tCO2',
                'Lagging_Current_Power_Factor': 'Power_Factor',
                'Leading_Current_Power_Factor': 'Leading_Power_Factor',
                'WeekStatus': 'Week_Status',
                'Day_of_week': 'Day_of_Week',
                'Day_of_Week': 'Day_of_Week'
                # Keep Usage_kWh, Load_Type, NSM as they are
            }
            
            # Apply column renaming only for columns that need renaming
            self.data = self.data.rename(columns=column_mapping)
            
            # Convert numeric columns to proper data types
            numeric_columns = ['Usage_kWh', 'Reactive_Power_kVarh', 'Leading_Reactive_Power_kVarh',
                              'CO2_tCO2', 'Power_Factor', 'Leading_Power_Factor', 'NSM']
            
            for col in numeric_columns:
                if col in self.data.columns:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            # Ensure categorical columns are properly typed
            categorical_columns = ['Week_Status', 'Day_of_Week', 'Load_Type']
            for col in categorical_columns:
                if col in self.data.columns:
                    # Convert to string to ensure consistent categorical handling
                    self.data[col] = self.data[col].astype(str)
            
            # Remove any duplicate columns that might have been created
            self.data = self.data.loc[:, ~self.data.columns.duplicated()]
            
            # Final cleanup - ensure only unique, properly named columns
            self.data = self.data.loc[:, ~self.data.columns.duplicated()]
            
            # Define expected final columns after processing
            expected_columns = [
                'Date', 'Usage_kWh', 'Reactive_Power_kVarh', 'Leading_Reactive_Power_kVarh',
                'CO2_tCO2', 'Power_Factor', 'Leading_Power_Factor', 'NSM', 
                'Week_Status', 'Day_of_Week', 'Load_Type'
            ]
            
            # Keep only expected columns that exist
            final_columns = [col for col in expected_columns if col in self.data.columns]
            self.data = self.data[final_columns]
            
            # If no Usage_kWh column, use the first numeric column as energy usage
            if 'Usage_kWh' not in self.data.columns:
                numeric_cols = self.data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    self.data['Usage_kWh'] = self.data[numeric_cols[0]]
                    print(f"Warning: No Usage_kWh column found, using {numeric_cols[0]} as energy usage")
            
            print(f"✅ Data processed successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
            print(f"Final clean columns: {list(self.data.columns)}")
            
            return self.data
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def clean_data(self, df):
        """Clean and preprocess the data"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Convert numeric columns to proper types first
        numeric_columns = ['Usage_kWh', 'Reactive_Power_kVarh', 'Leading_Reactive_Power_kVarh',
                          'CO2_tCO2', 'Power_Factor', 'Leading_Power_Factor', 'NSM']
        
        for col in numeric_columns:
            if col in df.columns:
                # Convert to numeric, errors='coerce' will turn invalid values to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle missing values for numeric columns only
        numeric_columns_present = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns_present] = df[numeric_columns_present].fillna(df[numeric_columns_present].mean())
        
        # Handle missing values for categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'Date':  # Don't fill Date column
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Remove extreme outliers (optional) - only for numeric columns
        for col in numeric_columns_present:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df
    
    def prepare_regression_data(self, df):
        """Prepare features and target for regression"""
        df = df.copy()
        
        # Create time-based features
        if 'Date' in df.columns:
            df['Day_of_Year'] = df['Date'].dt.dayofyear
            df['Hour'] = df['Date'].dt.hour
            df['Month'] = df['Date'].dt.month
            df['Quarter'] = df['Date'].dt.quarter
        
        # Create synthetic temperature and humidity from industrial data for compatibility
        if 'Power_Factor' in df.columns and 'Temperature' not in df.columns:
            # Use power factor as proxy for temperature (scaled and shifted)
            df['Temperature'] = 20 + (df['Power_Factor'] - df['Power_Factor'].mean()) * 10
        
        if 'CO2_tCO2' in df.columns and 'Humidity' not in df.columns:
            # Use CO2 as proxy for humidity (scaled and shifted)
            df['Humidity'] = 50 + (df['CO2_tCO2'] - df['CO2_tCO2'].mean()) * 5
        
        # Handle Day_of_Week - convert string to numeric
        if 'Day_of_Week' in df.columns:
            if df['Day_of_Week'].dtype == 'object':  # String values like 'Monday'
                day_mapping = {
                    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                    'Friday': 4, 'Saturday': 5, 'Sunday': 6
                }
                df['Day_of_Week'] = df['Day_of_Week'].map(day_mapping).fillna(0)
        elif 'Date' in df.columns:
            df['Day_of_Week'] = df['Date'].dt.dayofweek
        
        # Create lag features for energy usage
        df['Usage_lag1'] = df['Usage_kWh'].shift(1)
        df['Usage_lag2'] = df['Usage_kWh'].shift(2)
        df['Usage_lag3'] = df['Usage_kWh'].shift(3)
        
        # Create rolling features
        df['Usage_rolling_3'] = df['Usage_kWh'].rolling(window=3).mean()
        df['Usage_rolling_7'] = df['Usage_kWh'].rolling(window=7).mean()
        
        # Industrial-specific features
        available_features = []
        
        # Always include time-based features
        if 'Day_of_Year' in df.columns:
            available_features.append('Day_of_Year')
        if 'Day_of_Week' in df.columns:
            available_features.append('Day_of_Week')
        if 'Hour' in df.columns:
            available_features.append('Hour')
        if 'Month' in df.columns:
            available_features.append('Month')
        if 'Quarter' in df.columns:
            available_features.append('Quarter')
        
        # Add lag features
        available_features.extend(['Usage_lag1', 'Usage_lag2', 'Usage_lag3', 
                                 'Usage_rolling_3', 'Usage_rolling_7'])
        
        # Add steel industry specific features if available
        steel_industry_features = [
            'Reactive_Power_kVarh', 
            'Leading_Reactive_Power_kVarh',
            'CO2_tCO2', 
            'Power_Factor',
            'Leading_Power_Factor',
            'NSM'
        ]
        
        for feature in steel_industry_features:
            if feature in df.columns:
                available_features.append(feature)
        
        # Encode categorical features
        if 'Week_Status' in df.columns:
            # Convert to numeric if it's string
            if df['Week_Status'].dtype == 'object':
                # Handle both 'Weekend'/'Weekday' and 'Weekday'/'Weekend' patterns
                df['Is_Weekend'] = df['Week_Status'].str.contains('Weekend', case=False, na=False).astype(int)
            else:
                df['Is_Weekend'] = (df['Week_Status'] == 'Weekend').astype(int)
            available_features.append('Is_Weekend')
        
        if 'Load_Type' in df.columns:
            # One-hot encode load types
            load_dummies = pd.get_dummies(df['Load_Type'], prefix='Load')
            df = pd.concat([df, load_dummies], axis=1)
            available_features.extend(load_dummies.columns.tolist())
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Select only available features
        feature_columns = [col for col in available_features if col in df.columns]
        
        if len(feature_columns) == 0:
            raise ValueError("No valid features found for regression")
        
        X = df[feature_columns].values
        y = df['Usage_kWh'].values
        
        # Normalize features
        X = self.normalize_features(X)
        
        return X, y
    
    def prepare_clustering_features(self, df):
        """Prepare features for clustering analysis"""
        df = df.copy()
        
        # Create rolling window features
        df['Usage_rolling_3'] = df['Usage_kWh'].rolling(window=3, min_periods=1).mean()
        df['Usage_rolling_7'] = df['Usage_kWh'].rolling(window=7, min_periods=1).mean()
        df['Usage_std_3'] = df['Usage_kWh'].rolling(window=3, min_periods=1).std().fillna(0)
        df['Usage_std_7'] = df['Usage_kWh'].rolling(window=7, min_periods=1).std().fillna(0)
        
        # Time-based features
        clustering_features = ['Usage_kWh']
        
        if 'Date' in df.columns:
            df['Day_of_Year'] = df['Date'].dt.dayofyear
            df['Hour'] = df['Date'].dt.hour
            df['Month'] = df['Date'].dt.month
            clustering_features.extend(['Day_of_Year', 'Hour', 'Month'])
        
        # Handle Day_of_Week for clustering
        if 'Day_of_Week' in df.columns:
            if df['Day_of_Week'].dtype == 'object':  # String values like 'Monday'
                day_mapping = {
                    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                    'Friday': 4, 'Saturday': 5, 'Sunday': 6
                }
                df['Day_of_Week'] = df['Day_of_Week'].map(day_mapping).fillna(0)
            clustering_features.append('Day_of_Week')
        elif 'Date' in df.columns:
            df['Day_of_Week'] = df['Date'].dt.dayofweek
            clustering_features.append('Day_of_Week')
        
        # Add rolling features
        clustering_features.extend(['Usage_rolling_3', 'Usage_rolling_7', 'Usage_std_3', 'Usage_std_7'])
        
        # Steel industry features for clustering
        steel_clustering_features = [
            'Reactive_Power_kVarh',
            'Leading_Reactive_Power_kVarh', 
            'CO2_tCO2', 
            'Power_Factor',
            'Leading_Power_Factor',
            'NSM'
        ]
        
        for feature in steel_clustering_features:
            if feature in df.columns:
                clustering_features.append(feature)
        
        # Categorical features
        if 'Week_Status' in df.columns:
            # Convert to numeric if it's string
            if df['Week_Status'].dtype == 'object':
                # Handle both 'Weekend'/'Weekday' and 'Weekday'/'Weekend' patterns
                df['Is_Weekend'] = df['Week_Status'].str.contains('Weekend', case=False, na=False).astype(int)
            else:
                df['Is_Weekend'] = (df['Week_Status'] == 'Weekend').astype(int)
            clustering_features.append('Is_Weekend')
        
        if 'Load_Type' in df.columns:
            # Simple label encoding for clustering
            unique_types = df['Load_Type'].unique()
            type_mapping = {load_type: i for i, load_type in enumerate(unique_types)}
            df['Load_Type_Encoded'] = df['Load_Type'].map(type_mapping).astype(float)
            clustering_features.append('Load_Type_Encoded')
        
        # Select only available features
        available_features = [col for col in clustering_features if col in df.columns]
        
        if len(available_features) == 0:
            raise ValueError("No valid features found for clustering")
        
        # Convert to numpy array
        features = df[available_features].values
        
        # Handle any remaining NaN values
        features = np.nan_to_num(features, nan=0.0)
        
        # Normalize features
        features = self.normalize_features(features)
        
        return features
    
    def normalize_features(self, X):
        """Normalize features using min-max scaling"""
        # Convert to numpy array and ensure numeric type
        try:
            X = np.array(X, dtype=float)
        except ValueError as e:
            print(f"Error converting to float: {e}")
            # Try to identify non-numeric values
            if len(X.shape) == 2:
                for i in range(X.shape[1]):
                    try:
                        X[:, i] = pd.to_numeric(X[:, i], errors='coerce')
                    except:
                        print(f"Column {i} contains non-numeric values")
            X = np.nan_to_num(X, nan=0.0)  # Replace NaN with 0
        
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        
        # Handle constant features
        X_range = X_max - X_min
        X_range[X_range == 0] = 1
        
        X_normalized = (X - X_min) / X_range
        
        return X_normalized
    
    def create_time_windows(self, df, window_size=7):
        """Create sliding time windows for analysis"""
        windows = []
        
        for i in range(len(df) - window_size + 1):
            window = df.iloc[i:i + window_size]
            windows.append(window)
        
        return windows
    
    def calculate_statistics(self, df):
        """Calculate basic statistics for the dataset"""
        stats = {
            'mean': df['Usage_kWh'].mean(),
            'std': df['Usage_kWh'].std(),
            'min': df['Usage_kWh'].min(),
            'max': df['Usage_kWh'].max(),
            'median': df['Usage_kWh'].median(),
            'count': len(df)
        }
        
        return stats