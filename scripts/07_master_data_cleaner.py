import pandas as pd
import numpy as np
import os

def is_valid_country(name):
    # Exclude aggregates, funds, and non-country entities
    invalid_keywords = [
        'Fund', 'World', 'income', 'countries', 'Euro', 'IDA', 'IBRD', 'OECD', 'Arab', 'Sub-Saharan',
        'Europe', 'Asia', 'Africa', 'America', 'Caribbean', 'Pacific', 'East Asia', 'South Asia',
        'Middle East', 'North Africa', 'Latin America', 'High income', 'Low income', 'Upper middle income',
        'Lower middle income', 'All', 'Total', 'Other', 'Region', 'Area', 'States', 'UNDP', 'EMU', 'GCC',
        'LDC', 'LLDC', 'SIDS', 'Fragile', 'Small states', 'Least developed', 'Developing', 'Advanced', 'Emerging'
    ]
    if pd.isnull(name):
        return False
    for kw in invalid_keywords:
        if kw.lower() in name.lower():
            return False
    return True

def clean_and_combine_all_data():
    print("Loading base Economic Freedom + Gini Index data...")
    base = pd.read_csv('data/processed/economic_freedom_gini_combined.csv')
    base = base.rename(columns={'Index Year': 'Year'})
    base = base[base['Country'].apply(is_valid_country)]
    
    # Standardize country and year columns
    base['Country'] = base['Country'].str.strip()
    base['Year'] = pd.to_numeric(base['Year'], errors='coerce')
    
    # List of (filename, cleaning_function, new_column_names)
    merge_sources = [
        ('data/raw/unemployment_rate.csv', clean_unemployment_data, ['Unemployment_Rate']),
        ('data/raw/human_development_index.csv', clean_hdi_data, ['HDI']),
        ('data/raw/life_expectancy.csv', clean_life_expectancy_data, ['Life_Expectancy']),
        ('data/raw/extreme_poverty_share.csv', clean_poverty_data, ['Extreme_Poverty_Share']),
        ('data/raw/inflation_consumer_prices.csv', clean_inflation_data, ['Inflation_Rate']),
        ('data/raw/foreign_aid_received.csv', lambda df: clean_aid_data(df, 'aid_received'), ['Foreign_Aid_Received']),
        ('data/raw/foreign_aid_given.csv', lambda df: clean_aid_data(df, 'aid_given'), ['Foreign_Aid_Given']),
        ('data/raw/gdp_per_capita_worldbank.xlsx', clean_gdp_data, ['GDP_per_Capita'])
    ]
    
    master = base.copy()
    for path, clean_func, new_cols in merge_sources:
        print(f"Merging {os.path.basename(path)} ...")
        if path.endswith('.xlsx'):
            try:
                df = pd.read_excel(path)
            except Exception as e:
                print(f"  Could not load {path}: {e}")
                continue
        else:
            df = pd.read_csv(path)
        cleaned = clean_func(df)
        print(f"  Columns after cleaning: {list(cleaned.columns)}")
        if 'Country' not in cleaned.columns or 'Year' not in cleaned.columns:
            print(f"  Skipping {path} because 'Country' or 'Year' column is missing after cleaning.")
            continue
        if cleaned.empty:
            print(f"  Skipping {path} because cleaned DataFrame is empty.")
            continue
        print(cleaned.head())
        cleaned['Country'] = cleaned['Country'].astype(str).str.strip()
        cleaned['Year'] = pd.to_numeric(cleaned['Year'], errors='coerce')
        cleaned = cleaned[cleaned['Country'].apply(is_valid_country)]
        # Only keep columns needed for merge
        merge_cols = ['Country', 'Year'] + new_cols
        cleaned = cleaned[merge_cols]
        master = pd.merge(master, cleaned, on=['Country', 'Year'], how='left')
    
    # Save
    output_path = 'data/processed/master_dataset.csv'
    master.to_csv(output_path, index=False)
    print(f"\nSaved cleaned master dataset to {output_path}")
    print(f"Rows: {len(master)}, Columns: {list(master.columns)}")
    return master

# Cleaning functions for each dataset

def clean_unemployment_data(df):
    df = df.rename(columns={
        'Entity': 'Country',
        'Unemployment, total (% of total labor force) (modeled ILO estimate)': 'Unemployment_Rate'
    })
    return df[['Country', 'Year', 'Unemployment_Rate']]

def clean_hdi_data(df):
    df = df.rename(columns={'Entity': 'Country', 'Human Development Index': 'HDI'})
    return df[['Country', 'Year', 'HDI']]

def clean_life_expectancy_data(df):
    df = df.rename(columns={
        'Entity': 'Country',
        'Period life expectancy at birth - Sex: total - Age: 0': 'Life_Expectancy'
    })
    return df[['Country', 'Year', 'Life_Expectancy']]

def clean_poverty_data(df):
    df = df.rename(columns={'Country': 'Country', 'Share below $2.15 a day': 'Extreme_Poverty_Share'})
    return df[['Country', 'Year', 'Extreme_Poverty_Share']]

def clean_inflation_data(df):
    df = df.rename(columns={
        'Entity': 'Country',
        'Inflation, consumer prices (annual %)': 'Inflation_Rate'
    })
    return df[['Country', 'Year', 'Inflation_Rate']]

def clean_aid_data(df, aid_type):
    value_col = df.columns[3]
    colname = 'Foreign_Aid_Received' if aid_type == 'aid_received' else 'Foreign_Aid_Given'
    df = df.rename(columns={'Entity': 'Country', value_col: colname})
    return df[['Country', 'Year', colname]]

def clean_gdp_data(df):
    # Try to find the right columns (World Bank format)
    if 'Country Name' in df.columns:
        id_cols = ['Country Name', 'Country Code']
        value_cols = [col for col in df.columns if col.isdigit()]
        df_long = df.melt(id_vars=id_cols, value_vars=value_cols, var_name='Year', value_name='GDP_per_Capita')
        df_long = df_long.rename(columns={'Country Name': 'Country'})
        df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce')
        return df_long[['Country', 'Year', 'GDP_per_Capita']]
    # If already long format
    if {'Country', 'Year', 'GDP_per_Capita'}.issubset(df.columns):
        return df[['Country', 'Year', 'GDP_per_Capita']]
    # Otherwise, return empty
    return pd.DataFrame(columns=['Country', 'Year', 'GDP_per_Capita'])

if __name__ == "__main__":
    clean_and_combine_all_data() 