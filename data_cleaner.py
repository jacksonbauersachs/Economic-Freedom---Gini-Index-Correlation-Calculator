import pandas as pd
import numpy as np

def clean_and_combine_data():
    """
    Clean and combine Heritage Index and Gini Index datasets.
    Only includes data where both datasets have information for the same country and year.
    """
    
    print("Loading Heritage Index data...")
    # Load Heritage Index data - skip the first 4 lines which are headers
    heritage_df = pd.read_csv('heritage-index-of-economic-freedom-20250620205426.csv', skiprows=4)
    
    print("Loading Gini Index data...")
    # Load Gini Index data
    gini_df = pd.read_csv('API_SI.POV.GINI_DS2_en_csv_v2_81093.csv', skiprows=4)
    
    print("Cleaning Heritage Index data...")
    # Clean Heritage Index data
    # Remove rows where Overall Score is N/A
    heritage_df = heritage_df[heritage_df['Overall Score'] != 'N/A']
    
    # Convert Overall Score to numeric
    heritage_df['Overall Score'] = pd.to_numeric(heritage_df['Overall Score'], errors='coerce')
    
    # Remove rows where Overall Score is NaN
    heritage_df = heritage_df.dropna(subset=['Overall Score'])
    
    # Convert Index Year to int
    heritage_df['Index Year'] = heritage_df['Index Year'].astype(int)
    
    print("Cleaning Gini Index data...")
    # Clean Gini Index data
    # Remove rows where Country Code is empty or is a region (not a country)
    gini_df = gini_df[gini_df['Country Code'].notna()]
    
    # Remove regional aggregations (these have 3-letter codes that are not country codes)
    # Keep only rows where Country Code is 3 letters and not a region
    regions_to_exclude = [
        'AFE', 'AFW', 'ARB', 'CEB', 'CSS', 'EAP', 'EAR', 'EAS', 'ECA', 'ECS', 
        'EMU', 'EUU', 'FCS', 'HIC', 'HPC', 'IBD', 'IBT', 'IDA', 'IDB', 'IDX',
        'LAC', 'LCN', 'LDC', 'LIC', 'LMC', 'LMY', 'LTE', 'MEA', 'MIC', 'MNA',
        'NAC', 'OED', 'OSS', 'PSS', 'PST', 'SAS', 'SSA', 'SSF', 'SST', 'TEA',
        'TEC', 'TLA', 'TMN', 'TSA', 'TSS'
    ]
    gini_df = gini_df[~gini_df['Country Code'].isin(regions_to_exclude)]
    
    # Melt the Gini data to convert from wide to long format
    # This will create rows for each country-year combination
    gini_melted = gini_df.melt(
        id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
        var_name='Year',
        value_name='Gini_Index'
    )
    
    # Convert Year to int and filter to reasonable years (1995-2024)
    gini_melted['Year'] = pd.to_numeric(gini_melted['Year'], errors='coerce')
    gini_melted = gini_melted.dropna(subset=['Year'])
    gini_melted = gini_melted[(gini_melted['Year'] >= 1995) & (gini_melted['Year'] <= 2024)]
    
    # Convert Gini_Index to numeric
    gini_melted['Gini_Index'] = pd.to_numeric(gini_melted['Gini_Index'], errors='coerce')
    
    # Remove rows where Gini_Index is NaN
    gini_melted = gini_melted.dropna(subset=['Gini_Index'])
    
    print("Matching countries between datasets...")
    # Create a mapping for country names that might differ between datasets
    country_mapping = {
        # Heritage Index name -> Gini Index name
        'Bahamas, The': 'Bahamas, The',
        'Congo, Dem. Rep.': 'Congo, Dem. Rep.',
        'Congo, Rep.': 'Congo, Rep.',
        'Cote d\'Ivoire': 'Cote d\'Ivoire',
        'Egypt, Arab Rep.': 'Egypt, Arab Rep.',
        'Gambia, The': 'Gambia, The',
        'Hong Kong SAR, China': 'Hong Kong SAR, China',
        'Iran, Islamic Rep.': 'Iran, Islamic Rep.',
        'Korea, Dem. People\'s Rep.': 'Korea, Dem. People\'s Rep.',
        'Korea, Rep.': 'Korea, Rep.',
        'Kyrgyz Republic': 'Kyrgyz Republic',
        'Lao PDR': 'Lao PDR',
        'Macao SAR, China': 'Macao SAR, China',
        'Micronesia, Fed. Sts.': 'Micronesia, Fed. Sts.',
        'North Macedonia': 'North Macedonia',
        'Russian Federation': 'Russian Federation',
        'Slovak Republic': 'Slovak Republic',
        'Syrian Arab Republic': 'Syrian Arab Republic',
        'Turkiye': 'Turkiye',
        'United Arab Emirates': 'United Arab Emirates',
        'United Kingdom': 'United Kingdom',
        'United States': 'United States',
        'Venezuela, RB': 'Venezuela, RB',
        'Viet Nam': 'Viet Nam',
        'Yemen, Rep.': 'Yemen, Rep.'
    }
    
    # Apply country name mapping to Heritage data
    heritage_df['Country_Mapped'] = heritage_df['Country'].map(lambda x: country_mapping.get(x, x))
    
    print("Merging datasets...")
    # Merge the datasets on country name and year
    merged_df = pd.merge(
        heritage_df,
        gini_melted[['Country Name', 'Year', 'Gini_Index']],
        left_on=['Country_Mapped', 'Index Year'],
        right_on=['Country Name', 'Year'],
        how='inner'
    )
    
    # Clean up the merged dataset
    cols_to_drop = [col for col in ['Country_Mapped', 'Country Name_y', 'Year'] if col in merged_df.columns]
    merged_df = merged_df.drop(cols_to_drop, axis=1)
    if 'Country Name_x' in merged_df.columns:
        merged_df = merged_df.rename(columns={'Country Name_x': 'Country'})
    
    # Reorder columns for better readability
    column_order = [
        'Country', 'Index Year', 'Overall Score', 'Gini_Index',
        'Property Rights', 'Government Integrity', 'Judicial Effectiveness',
        'Tax Burden', 'Government Spending', 'Fiscal Health',
        'Business Freedom', 'Labor Freedom', 'Monetary Freedom',
        'Trade Freedom', 'Investment Freedom', 'Financial Freedom'
    ]
    
    # Only include columns that exist in the merged dataset
    existing_columns = [col for col in column_order if col in merged_df.columns]
    merged_df = merged_df[existing_columns]
    
    # Sort by country and year
    merged_df = merged_df.sort_values(['Country', 'Index Year'])
    
    print(f"Final dataset shape: {merged_df.shape}")
    print(f"Number of unique countries: {merged_df['Country'].nunique()}")
    print(f"Year range: {merged_df['Index Year'].min()} - {merged_df['Index Year'].max()}")
    
    # Save the cleaned and combined dataset
    output_filename = 'economic_freedom_gini_combined.csv'
    merged_df.to_csv(output_filename, index=False)
    print(f"Combined dataset saved to: {output_filename}")
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Total observations: {len(merged_df)}")
    print(f"Countries with data: {merged_df['Country'].nunique()}")
    print(f"Years covered: {merged_df['Index Year'].min()} - {merged_df['Index Year'].max()}")
    
    # Show sample of countries with most data points
    country_counts = merged_df['Country'].value_counts().head(10)
    print(f"\nTop 10 countries by number of observations:")
    for country, count in country_counts.items():
        print(f"  {country}: {count} observations")
    
    return merged_df

if __name__ == "__main__":
    combined_data = clean_and_combine_data() 