import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calculate_correlation_with_stats(x, y):
    """
    Calculate correlation coefficient, p-value, and sample size.
    Returns (correlation, p_value, sample_size) or (None, None, 0) if insufficient data.
    """
    # Remove NaN values
    valid_data = pd.DataFrame({'x': x, 'y': y}).dropna()
    
    if len(valid_data) < 50:  # Filter out correlations with less than 50 data points
        return None, None, 0
    
    if len(valid_data) == 0:
        return None, None, 0
    
    # Calculate correlation
    correlation, p_value = stats.pearsonr(valid_data['x'], valid_data['y'])
    
    return correlation, p_value, len(valid_data)

def calculate_confidence_interval(correlation, sample_size, confidence=0.95):
    """Calculate confidence interval for correlation coefficient."""
    if sample_size < 3:
        return None, None
    
    # Fisher's z transformation
    z = np.arctanh(correlation)
    
    # Standard error
    se = 1 / np.sqrt(sample_size - 3)
    
    # Critical value (z-score for confidence level)
    if confidence == 0.95:
        critical_value = 1.96
    elif confidence == 0.99:
        critical_value = 2.576
    else:
        critical_value = 1.96
    
    # Confidence interval in z-space
    z_lower = z - critical_value * se
    z_upper = z + critical_value * se
    
    # Transform back to correlation space
    ci_lower = np.tanh(z_lower)
    ci_upper = np.tanh(z_upper)
    
    return ci_lower, ci_upper

def interpret_correlation_strength(correlation):
    """Interpret correlation strength."""
    if correlation is None:
        return "Insufficient Data"
    
    abs_corr = abs(correlation)
    if abs_corr >= 0.7:
        return "Very Strong"
    elif abs_corr >= 0.5:
        return "Strong"
    elif abs_corr >= 0.3:
        return "Moderate"
    elif abs_corr >= 0.1:
        return "Weak"
    else:
        return "Very Weak"

def single_table_correlation_analysis():
    """
    Perform correlation analysis and create ONE big table with all results.
    """
    print("Loading master dataset...")
    df = pd.read_csv('data/processed/master_dataset.csv')
    
    # Define economic freedom indicators
    economic_freedom_indicators = [
        'Overall Score',
        'Property Rights',
        'Government Integrity', 
        'Judicial Effectiveness',
        'Tax Burden',
        'Government Spending',
        'Fiscal Health',
        'Business Freedom',
        'Labor Freedom',
        'Monetary Freedom',
        'Trade Freedom',
        'Investment Freedom',
        'Financial Freedom'
    ]
    
    # Define outcome variables
    outcome_variables = [
        'Gini_Index',
        'Unemployment_Rate',
        'HDI',
        'Life_Expectancy',
        'Extreme_Poverty_Share',
        'Inflation_Rate',
        'Foreign_Aid_Received',
        'Foreign_Aid_Given'
    ]
    
    # Define lags to test
    lags = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]
    
    # Results storage
    all_results = []
    
    print("Calculating correlations...")
    
    # Calculate correlations for each economic freedom indicator and lag
    for ef_indicator in economic_freedom_indicators:
        for lag in lags:
            print(f"  Processing {ef_indicator} with {lag}-year lag...")
            
            # Create lagged data if needed
            if lag == 0:
                df_lagged = df.copy()
                ef_data = df_lagged[ef_indicator]
            else:
                df_lagged = df.copy()
                ef_data = df_lagged.groupby('Country')[ef_indicator].shift(lag)
            
            # Create row label
            if lag == 0:
                row_label = ef_indicator
            else:
                row_label = f"{ef_indicator} ({lag}-year lag)"
            
            # Initialize row data
            row_data = {
                'Economic_Freedom_Indicator': row_label,
                'Lag_Years': lag
            }
            
            # Calculate correlations for each outcome variable
            for outcome in outcome_variables:
                correlation, p_value, sample_size = calculate_correlation_with_stats(
                    ef_data, df_lagged[outcome]
                )
                
                if correlation is not None:
                    ci_lower, ci_upper = calculate_confidence_interval(correlation, sample_size)
                    strength = interpret_correlation_strength(correlation)
                    significance = "Significant" if p_value < 0.05 else "Not Significant"
                else:
                    correlation = np.nan
                    p_value = np.nan
                    sample_size = 0
                    ci_lower = np.nan
                    ci_upper = np.nan
                    strength = "Insufficient Data"
                    significance = "Insufficient Data"
                
                # Add all statistics for this outcome variable
                row_data[f'{outcome}_Correlation'] = correlation
                row_data[f'{outcome}_P_Value'] = p_value
                row_data[f'{outcome}_Sample_Size'] = sample_size
                row_data[f'{outcome}_CI_Lower'] = ci_lower
                row_data[f'{outcome}_CI_Upper'] = ci_upper
                row_data[f'{outcome}_Strength'] = strength
                row_data[f'{outcome}_Significance'] = significance
            
            all_results.append(row_data)
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Reorder columns to group by statistic type instead of outcome variable
    # First: Economic_Freedom_Indicator, Lag_Years
    # Then: All correlations, then all p-values, then all sample sizes, etc.
    base_columns = ['Economic_Freedom_Indicator', 'Lag_Years']
    
    # Get all outcome variables
    outcome_variables = [
        'Gini_Index',
        'Unemployment_Rate',
        'HDI',
        'Life_Expectancy',
        'Extreme_Poverty_Share',
        'Inflation_Rate',
        'Foreign_Aid_Received',
        'Foreign_Aid_Given'
    ]
    
    # Create new column order
    new_column_order = base_columns.copy()
    
    # Add all correlations first
    for outcome in outcome_variables:
        new_column_order.append(f'{outcome}_Correlation')
    
    # Add all p-values
    for outcome in outcome_variables:
        new_column_order.append(f'{outcome}_P_Value')
    
    # Add all sample sizes
    for outcome in outcome_variables:
        new_column_order.append(f'{outcome}_Sample_Size')
    
    # Add all confidence intervals
    for outcome in outcome_variables:
        new_column_order.append(f'{outcome}_CI_Lower')
        new_column_order.append(f'{outcome}_CI_Upper')
    
    # Add all strengths
    for outcome in outcome_variables:
        new_column_order.append(f'{outcome}_Strength')
    
    # Add all significance
    for outcome in outcome_variables:
        new_column_order.append(f'{outcome}_Significance')
    
    # Reorder the DataFrame
    results_df = results_df[new_column_order]
    
    # Round numeric columns
    numeric_columns = [col for col in results_df.columns if any(x in col for x in ['Correlation', 'P_Value', 'CI_Lower', 'CI_Upper'])]
    results_df[numeric_columns] = results_df[numeric_columns].round(4)
    
    # Sort by economic freedom indicator and lag
    results_df = results_df.sort_values(['Economic_Freedom_Indicator', 'Lag_Years'])
    
    # Save results
    print("Saving single table results...")
    results_df.to_csv('data/results/single_table_correlation_analysis.csv', index=False)
    
    # Create Excel file
    with pd.ExcelWriter('data/results/single_table_correlation_analysis.xlsx', engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='All_Correlations', index=False)
        
        # Also create a summary sheet
        summary_data = []
        for ef_indicator in economic_freedom_indicators:
            for lag in lags:
                if lag == 0:
                    row_label = ef_indicator
                else:
                    row_label = f"{ef_indicator} ({lag}-year lag)"
                
                # Get row data
                row = results_df[results_df['Economic_Freedom_Indicator'] == row_label].iloc[0]
                
                # Calculate summary stats
                correlations = [row[f'{var}_Correlation'] for var in outcome_variables if not pd.isna(row[f'{var}_Correlation'])]
                p_values = [row[f'{var}_P_Value'] for var in outcome_variables if not pd.isna(row[f'{var}_P_Value'])]
                significant_count = len([p for p in p_values if p < 0.05])
                
                summary_data.append({
                    'Economic_Freedom_Indicator': row_label,
                    'Lag_Years': lag,
                    'Total_Outcomes': len(correlations),
                    'Significant_Correlations': significant_count,
                    'Strong_Correlations': len([c for c in correlations if abs(c) >= 0.5]),
                    'Max_Correlation': max(correlations) if correlations else np.nan,
                    'Min_Correlation': min(correlations) if correlations else np.nan,
                    'Avg_Correlation': np.mean(correlations) if correlations else np.nan,
                    'Max_P_Value': max(p_values) if p_values else np.nan,
                    'Min_P_Value': min(p_values) if p_values else np.nan
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    # Print summary
    print(f"\n=== SINGLE TABLE CORRELATION ANALYSIS COMPLETE ===")
    print(f"Economic freedom indicators analyzed: {len(economic_freedom_indicators)}")
    print(f"Outcome variables analyzed: {len(outcome_variables)}")
    print(f"Lags tested: {len(lags)}")
    print(f"Total rows in table: {len(results_df)}")
    print(f"Total columns in table: {len(results_df.columns)}")
    
    # Show strongest correlations
    print(f"\n=== TOP 10 STRONGEST CORRELATIONS ===")
    all_correlations = []
    for idx, row in results_df.iterrows():
        for outcome in outcome_variables:
            correlation = row[f'{outcome}_Correlation']
            p_value = row[f'{outcome}_P_Value']
            sample_size = row[f'{outcome}_Sample_Size']
            
            if not pd.isna(correlation):
                all_correlations.append({
                    'Indicator': row['Economic_Freedom_Indicator'],
                    'Outcome': outcome,
                    'Correlation': correlation,
                    'P_Value': p_value,
                    'Sample_Size': sample_size
                })
    
    # Sort by absolute correlation
    all_correlations.sort(key=lambda x: abs(x['Correlation']), reverse=True)
    
    for i, corr in enumerate(all_correlations[:10]):
        significance = "***" if corr['P_Value'] < 0.001 else "**" if corr['P_Value'] < 0.01 else "*" if corr['P_Value'] < 0.05 else ""
        print(f"{i+1}. {corr['Indicator']} → {corr['Outcome']}: r = {corr['Correlation']:.4f}{significance} (n = {corr['Sample_Size']}, p = {corr['P_Value']:.4f})")
    
    print(f"\nResults saved to:")
    print(f"- data/results/single_table_correlation_analysis.csv")
    print(f"- data/results/single_table_correlation_analysis.xlsx")
    
    # Show table structure
    print(f"\n=== TABLE STRUCTURE ===")
    print(f"Columns: {list(results_df.columns)}")
    print(f"Sample row:")
    print(results_df.iloc[0].to_dict())

if __name__ == "__main__":
    single_table_correlation_analysis() 