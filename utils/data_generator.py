import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_energy_data(output_path, n_days=365, start_date='2023-01-01'):
    """Generate synthetic energy consumption data"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate date range
    start = datetime.strptime(start_date, '%Y-%m-%d')
    dates = [start + timedelta(days=i) for i in range(n_days)]
    
    # Generate base patterns
    data = []
    
    for i, date in enumerate(dates):
        # Seasonal pattern (higher usage in winter/summer)
        day_of_year = date.timetuple().tm_yday
        seasonal_factor = 1 + 0.3 * np.cos(2 * np.pi * day_of_year / 365.25)
        
        # Weekly pattern (higher usage on weekdays)
        day_of_week = date.weekday()
        weekly_factor = 1.2 if day_of_week < 5 else 0.8  # Higher on weekdays
        
        # Temperature simulation (affects energy usage)
        temp_base = 20 + 15 * np.sin(2 * np.pi * day_of_year / 365.25)  # Seasonal temp
        temperature = temp_base + np.random.normal(0, 5)  # Add noise
        
        # Humidity simulation
        humidity = 50 + 20 * np.sin(2 * np.pi * day_of_year / 365.25 + np.pi/4) + np.random.normal(0, 10)
        humidity = np.clip(humidity, 20, 90)  # Keep within realistic range
        
        # Base energy usage (affected by temperature)
        temp_effect = 1 + 0.02 * abs(temperature - 22)  # Higher usage when temp deviates from 22°C
        
        # Base energy consumption
        base_usage = 50 * seasonal_factor * weekly_factor * temp_effect
        
        # Add some random noise
        noise = np.random.normal(0, 5)
        
        # Final usage
        usage = base_usage + noise
        
        # Ensure usage is positive
        usage = max(usage, 10)
        
        # Add some random outliers (5% chance)
        if np.random.random() < 0.05:
            if np.random.random() < 0.5:
                usage *= 1.8  # High outlier
            else:
                usage *= 0.3  # Low outlier
        
        data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Usage_kWh': round(usage, 2),
            'Temperature': round(temperature, 1),
            'Humidity': round(humidity, 1),
            'Day_of_Week': day_of_week,
            'Month': date.month,
            'Season': get_season(date.month)
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(df)} records of energy data and saved to {output_path}")
    print(f"Usage range: {df['Usage_kWh'].min():.2f} - {df['Usage_kWh'].max():.2f} kWh")
    print(f"Temperature range: {df['Temperature'].min():.1f} - {df['Temperature'].max():.1f}°C")
    
    return df

def get_season(month):
    """Get season based on month"""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

def generate_industrial_energy_data(output_path, n_days=365):
    """Generate more complex industrial energy consumption data"""
    
    np.random.seed(42)
    
    # Generate dates
    start = datetime(2023, 1, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    
    data = []
    
    # Define different operating modes
    operating_modes = {
        'normal': {'base': 100, 'variance': 10},
        'high_production': {'base': 150, 'variance': 15},
        'maintenance': {'base': 30, 'variance': 5},
        'weekend': {'base': 60, 'variance': 8}
    }
    
    for i, date in enumerate(dates):
        # Determine operating mode
        day_of_week = date.weekday()
        
        if day_of_week >= 5:  # Weekend
            mode = 'weekend'
        elif np.random.random() < 0.1:  # 10% chance of maintenance
            mode = 'maintenance'
        elif np.random.random() < 0.2:  # 20% chance of high production
            mode = 'high_production'
        else:
            mode = 'normal'
        
        # Get mode parameters
        base_usage = operating_modes[mode]['base']
        variance = operating_modes[mode]['variance']
        
        # Environmental factors
        day_of_year = date.timetuple().tm_yday
        temperature = 20 + 15 * np.sin(2 * np.pi * day_of_year / 365.25) + np.random.normal(0, 3)
        humidity = 50 + 20 * np.cos(2 * np.pi * day_of_year / 365.25) + np.random.normal(0, 8)
        humidity = np.clip(humidity, 20, 90)
        
        # Energy usage calculation
        temp_factor = 1 + 0.01 * abs(temperature - 20)
        seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * day_of_year / 365.25)
        
        usage = base_usage * temp_factor * seasonal_factor + np.random.normal(0, variance)
        usage = max(usage, 5)  # Minimum usage
        
        # Add equipment-specific patterns
        equipment_load = simulate_equipment_load(i)
        usage += equipment_load
        
        # Add rare extreme events (1% chance)
        if np.random.random() < 0.01:
            usage *= np.random.uniform(2.0, 3.0)  # Extreme high usage
        
        data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Usage_kWh': round(usage, 2),
            'Temperature': round(temperature, 1),
            'Humidity': round(humidity, 1),
            'Operating_Mode': mode,
            'Day_of_Week': day_of_week,
            'Month': date.month,
            'Equipment_Load': round(equipment_load, 2)
        })
    
    df = pd.DataFrame(data)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(df)} records of industrial energy data")
    return df

def simulate_equipment_load(day_index):
    """Simulate equipment-specific load patterns"""
    # Simulate different equipment cycles
    equipment_cycles = [
        5 * np.sin(2 * np.pi * day_index / 7),      # Weekly cycle
        3 * np.sin(2 * np.pi * day_index / 30),     # Monthly cycle
        2 * np.sin(2 * np.pi * day_index / 90)      # Quarterly cycle
    ]
    
    total_load = sum(equipment_cycles) + np.random.normal(0, 2)
    return max(total_load, 0)

if __name__ == "__main__":
    # Generate sample data
    generate_energy_data("data/energy_data.csv")
    generate_industrial_energy_data("data/industrial_energy_data.csv")