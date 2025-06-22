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

def formatted_correlation_analysis():
    """
    Perform correlation analysis and format results with economic freedom indicators as rows
    and outcome variables as columns, with separate sections for different statistics.
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
    correlation_data = []
    pvalue_data = []
    sample_size_data = []
    ci_lower_data = []
    ci_upper_data = []
    strength_data = []
    
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
            
            # Calculate correlations for each outcome variable
            row_correlations = []
            row_pvalues = []
            row_sample_sizes = []
            row_ci_lower = []
            row_ci_upper = []
            row_strengths = []
            
            for outcome in outcome_variables:
                correlation, p_value, sample_size = calculate_correlation_with_stats(
                    ef_data, df_lagged[outcome]
                )
                
                if correlation is not None:
                    ci_lower, ci_upper = calculate_confidence_interval(correlation, sample_size)
                    strength = interpret_correlation_strength(correlation)
                else:
                    correlation = np.nan
                    p_value = np.nan
                    sample_size = 0
                    ci_lower = np.nan
                    ci_upper = np.nan
                    strength = "Insufficient Data"
                
                row_correlations.append(correlation)
                row_pvalues.append(p_value)
                row_sample_sizes.append(sample_size)
                row_ci_lower.append(ci_lower)
                row_ci_upper.append(ci_upper)
                row_strengths.append(strength)
            
            # Create row label
            if lag == 0:
                row_label = ef_indicator
            else:
                row_label = f"{ef_indicator} ({lag}-year lag)"
            
            # Store results
            correlation_data.append([row_label] + row_correlations)
            pvalue_data.append([row_label] + row_pvalues)
            sample_size_data.append([row_label] + row_sample_sizes)
            ci_lower_data.append([row_label] + row_ci_lower)
            ci_upper_data.append([row_label] + row_ci_upper)
            strength_data.append([row_label] + row_strengths)
    
    # Create DataFrames
    columns = ['Economic_Freedom_Indicator'] + outcome_variables
    
    correlations_df = pd.DataFrame(correlation_data, columns=columns)
    pvalues_df = pd.DataFrame(pvalue_data, columns=columns)
    sample_sizes_df = pd.DataFrame(sample_size_data, columns=columns)
    ci_lower_df = pd.DataFrame(ci_lower_data, columns=columns)
    ci_upper_df = pd.DataFrame(ci_upper_data, columns=columns)
    strengths_df = pd.DataFrame(strength_data, columns=columns)
    
    # Round numeric values
    correlations_df[outcome_variables] = correlations_df[outcome_variables].round(4)
    pvalues_df[outcome_variables] = pvalues_df[outcome_variables].round(4)
    ci_lower_df[outcome_variables] = ci_lower_df[outcome_variables].round(4)
    ci_upper_df[outcome_variables] = ci_upper_df[outcome_variables].round(4)
    
    # Save results
    print("Saving formatted results...")
    
    # Create Excel file with multiple sheets
    with pd.ExcelWriter('data/results/formatted_correlation_analysis.xlsx', engine='openpyxl') as writer:
        # Correlations
        correlations_df.to_excel(writer, sheet_name='Correlations', index=False)
        
        # P-values (most important)
        pvalues_df.to_excel(writer, sheet_name='P_Values', index=False)
        
        # Sample sizes
        sample_sizes_df.to_excel(writer, sheet_name='Sample_Sizes', index=False)
        
        # Confidence intervals
        ci_lower_df.to_excel(writer, sheet_name='CI_Lower', index=False)
        ci_upper_df.to_excel(writer, sheet_name='CI_Upper', index=False)
        
        # Correlation strengths
        strengths_df.to_excel(writer, sheet_name='Correlation_Strengths', index=False)
        
        # Summary statistics
        summary_data = []
        for ef_indicator in economic_freedom_indicators:
            for lag in lags:
                if lag == 0:
                    row_label = ef_indicator
                else:
                    row_label = f"{ef_indicator} ({lag}-year lag)"
                
                # Get row data
                corr_row = correlations_df[correlations_df['Economic_Freedom_Indicator'] == row_label].iloc[0]
                pval_row = pvalues_df[pvalues_df['Economic_Freedom_Indicator'] == row_label].iloc[0]
                size_row = sample_sizes_df[sample_sizes_df['Economic_Freedom_Indicator'] == row_label].iloc[0]
                
                # Calculate summary stats
                valid_correlations = [corr_row[var] for var in outcome_variables if not pd.isna(corr_row[var])]
                significant_correlations = [pval_row[var] for var in outcome_variables if not pd.isna(pval_row[var]) and pval_row[var] < 0.05]
                
                summary_data.append({
                    'Economic_Freedom_Indicator': row_label,
                    'Total_Outcomes': len(valid_correlations),
                    'Significant_Correlations': len(significant_correlations),
                    'Strong_Correlations': len([c for c in valid_correlations if abs(c) >= 0.5]),
                    'Max_Correlation': max(valid_correlations) if valid_correlations else np.nan,
                    'Min_Correlation': min(valid_correlations) if valid_correlations else np.nan,
                    'Avg_Correlation': np.mean(valid_correlations) if valid_correlations else np.nan
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
    
    # Also save as CSV for easy viewing
    correlations_df.to_csv('data/results/formatted_correlations.csv', index=False)
    pvalues_df.to_csv('data/results/formatted_pvalues.csv', index=False)
    sample_sizes_df.to_csv('data/results/formatted_sample_sizes.csv', index=False)
    
    # Print summary
    print(f"\n=== FORMATTED CORRELATION ANALYSIS COMPLETE ===")
    print(f"Economic freedom indicators analyzed: {len(economic_freedom_indicators)}")
    print(f"Outcome variables analyzed: {len(outcome_variables)}")
    print(f"Lags tested: {len(lags)}")
    print(f"Total correlation combinations: {len(correlations_df)}")
    
    # Show strongest correlations
    print(f"\n=== TOP 10 STRONGEST CORRELATIONS ===")
    all_correlations = []
    for idx, row in correlations_df.iterrows():
        for outcome in outcome_variables:
            if not pd.isna(row[outcome]):
                all_correlations.append({
                    'Indicator': row['Economic_Freedom_Indicator'],
                    'Outcome': outcome,
                    'Correlation': row[outcome],
                    'P_Value': pvalues_df.loc[idx, outcome],
                    'Sample_Size': sample_sizes_df.loc[idx, outcome]
                })
    
    # Sort by absolute correlation
    all_correlations.sort(key=lambda x: abs(x['Correlation']), reverse=True)
    
    for i, corr in enumerate(all_correlations[:10]):
        significance = "***" if corr['P_Value'] < 0.001 else "**" if corr['P_Value'] < 0.01 else "*" if corr['P_Value'] < 0.05 else ""
        print(f"{i+1}. {corr['Indicator']} → {corr['Outcome']}: r = {corr['Correlation']:.4f}{significance} (n = {corr['Sample_Size']}, p = {corr['P_Value']:.4f})")
    
    print(f"\nResults saved to:")
    print(f"- data/results/formatted_correlation_analysis.xlsx (multiple sheets)")
    print(f"- data/results/formatted_correlations.csv")
    print(f"- data/results/formatted_pvalues.csv")
    print(f"- data/results/formatted_sample_sizes.csv")

if __name__ == "__main__":
    formatted_correlation_analysis() 