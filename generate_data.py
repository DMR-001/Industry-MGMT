#!/usr/bin/env python3
"""
Generate industrial energy data with steel industry compatible columns
"""

import sys
import os
sys.path.append('utils')

from data_generator import generate_industrial_energy_data

def main():
    """Generate the industrial energy data"""
    print("Generating industrial energy data with steel industry columns...")
    
    # Generate the data
    generate_industrial_energy_data('data/industrial_energy_data.csv')
    
    print("✅ Industrial energy data generated successfully!")
    print("This data includes steel industry compatible columns:")
    print("- Power_Factor")
    print("- CO2") 
    print("- Reactive_Power_kVarh")
    print("- Load_Type")
    print("- Week_Status")

if __name__ == "__main__":
    main()