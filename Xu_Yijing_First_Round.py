#!/usr/bin/env python
# coding: utf-8

"""
Yijing Xu First Round Interview
Data Processing and Analysis Script
"""

import pandas as pd
import numpy as np
import re
import os

# Read in the dataset
df = pd.read_csv('all_member_experience.csv')
print(df.shape)
print(df.head())

# Clean Date_from and Date_to columns
def extract_year(date_str):
    """
    Extracts the year from a date string.
    Returns 2023 if the date string is missing or NaN.
    """
    if pd.isna(date_str) or date_str == 'nan':
        return 2023  # Default year 2023 if no valid date is provided
    year_match = re.search(r'(19|20)\d{2}', date_str) 
    if year_match:
        return int(year_match.group(0))  
    else:
        return np.nan

df['year_from'] = df['date_from'].apply(extract_year)
df['year_to'] = df['date_to'].apply(extract_year)

# Remove rows with missing 'year_from'
df.dropna(subset=['year_from'], inplace=True)
print(f"Remaining rows after cleaning: {df.shape[0]}")

# Founder Identification
def is_founder(title):
    """
    Checks if the title indicates the person is a founder.
    Returns True if 'founder' is in the title.
    """
    if pd.isna(title):
        return False
    return 'founder' in title.lower()

df['is_founder'] = df['title'].apply(is_founder)
df_founder = df[df['is_founder']].sort_values(by=['member_id', 'year_from']).drop_duplicates(subset='member_id').reset_index()

# Save processed data to CSV
file_name = 'Founder.csv'
if os.path.isfile(file_name):
    os.remove(file_name)
df_founder.to_csv(file_name, index=True)

print("Data processing complete. Output saved to 'Founder.csv'")
