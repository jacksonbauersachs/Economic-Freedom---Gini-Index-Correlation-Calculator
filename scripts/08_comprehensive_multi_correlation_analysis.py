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

def comprehensive_correlation_analysis():
    """
    Perform comprehensive correlation analysis between economic freedom indicators
    and multiple outcome variables, including current and lagged correlations.
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
    
    # Results storage
    results = []
    
    print("Calculating current correlations...")
    
    # Current correlations (no lag)
    for ef_indicator in economic_freedom_indicators:
        for outcome in outcome_variables:
            correlation, p_value, sample_size = calculate_correlation_with_stats(
                df[ef_indicator], df[outcome]
            )
            
            if correlation is not None:
                ci_lower, ci_upper = calculate_confidence_interval(correlation, sample_size)
                strength = interpret_correlation_strength(correlation)
                significance = "Significant" if p_value < 0.05 else "Not Significant"
                
                results.append({
                    'Economic_Freedom_Indicator': ef_indicator,
                    'Outcome_Variable': outcome,
                    'Lag_Years': 0,
                    'Correlation': correlation,
                    'P_Value': p_value,
                    'Sample_Size': sample_size,
                    'CI_Lower': ci_lower,
                    'CI_Upper': ci_upper,
                    'Strength': strength,
                    'Significance': significance
                })
    
    print("Calculating lagged correlations...")
    
    # Lagged correlations (economic freedom predicts future outcomes)
    max_lag = 25
    for lag in range(1, max_lag + 1):
        print(f"  Processing lag {lag} years...")
        
        for ef_indicator in economic_freedom_indicators:
            for outcome in outcome_variables:
                # Create lagged variables
                df_lagged = df.copy()
                df_lagged[f'{ef_indicator}_lagged'] = df_lagged.groupby('Country')[ef_indicator].shift(lag)
                
                correlation, p_value, sample_size = calculate_correlation_with_stats(
                    df_lagged[f'{ef_indicator}_lagged'], df_lagged[outcome]
                )
                
                if correlation is not None:
                    ci_lower, ci_upper = calculate_confidence_interval(correlation, sample_size)
                    strength = interpret_correlation_strength(correlation)
                    significance = "Significant" if p_value < 0.05 else "Not Significant"
                    
                    results.append({
                        'Economic_Freedom_Indicator': ef_indicator,
                        'Outcome_Variable': outcome,
                        'Lag_Years': lag,
                        'Correlation': correlation,
                        'P_Value': p_value,
                        'Sample_Size': sample_size,
                        'CI_Lower': ci_lower,
                        'CI_Upper': ci_upper,
                        'Strength': strength,
                        'Significance': significance
                    })
    
    print("Calculating reverse lagged correlations...")
    
    # Reverse lagged correlations (outcomes predict future economic freedom)
    for lag in range(1, max_lag + 1):
        print(f"  Processing reverse lag {lag} years...")
        
        for ef_indicator in economic_freedom_indicators:
            for outcome in outcome_variables:
                # Create lagged variables
                df_lagged = df.copy()
                df_lagged[f'{outcome}_lagged'] = df_lagged.groupby('Country')[outcome].shift(lag)
                
                correlation, p_value, sample_size = calculate_correlation_with_stats(
                    df_lagged[f'{outcome}_lagged'], df_lagged[ef_indicator]
                )
                
                if correlation is not None:
                    ci_lower, ci_upper = calculate_confidence_interval(correlation, sample_size)
                    strength = interpret_correlation_strength(correlation)
                    significance = "Significant" if p_value < 0.05 else "Not Significant"
                    
                    results.append({
                        'Economic_Freedom_Indicator': ef_indicator,
                        'Outcome_Variable': outcome,
                        'Lag_Years': -lag,  # Negative to indicate reverse lag
                        'Correlation': correlation,
                        'P_Value': p_value,
                        'Sample_Size': sample_size,
                        'CI_Lower': ci_lower,
                        'CI_Upper': ci_upper,
                        'Strength': strength,
                        'Significance': significance
                    })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Round numeric columns
    numeric_columns = ['Correlation', 'P_Value', 'CI_Lower', 'CI_Upper']
    results_df[numeric_columns] = results_df[numeric_columns].round(4)
    
    # Sort by absolute correlation strength
    results_df['Abs_Correlation'] = abs(results_df['Correlation'])
    results_df = results_df.sort_values('Abs_Correlation', ascending=False)
    results_df = results_df.drop('Abs_Correlation', axis=1)
    
    # Save results
    print("Saving results...")
    results_df.to_csv('data/results/comprehensive_correlation_analysis.csv', index=False)
    
    # Create Excel file with multiple sheets
    with pd.ExcelWriter('data/results/comprehensive_correlation_analysis.xlsx', engine='openpyxl') as writer:
        # All results
        results_df.to_excel(writer, sheet_name='All_Results', index=False)
        
        # Current correlations only
        current_results = results_df[results_df['Lag_Years'] == 0].copy()
        current_results.to_excel(writer, sheet_name='Current_Correlations', index=False)
        
        # Forward lagged correlations (economic freedom predicts outcomes)
        forward_lagged = results_df[results_df['Lag_Years'] > 0].copy()
        forward_lagged.to_excel(writer, sheet_name='Forward_Lagged', index=False)
        
        # Reverse lagged correlations (outcomes predict economic freedom)
        reverse_lagged = results_df[results_df['Lag_Years'] < 0].copy()
        reverse_lagged.to_excel(writer, sheet_name='Reverse_Lagged', index=False)
        
        # Significant correlations only
        significant_results = results_df[results_df['Significance'] == 'Significant'].copy()
        significant_results.to_excel(writer, sheet_name='Significant_Only', index=False)
        
        # Strong correlations only (|r| >= 0.5)
        strong_results = results_df[abs(results_df['Correlation']) >= 0.5].copy()
        strong_results.to_excel(writer, sheet_name='Strong_Correlations', index=False)
    
    # Print summary
    print(f"\n=== COMPREHENSIVE CORRELATION ANALYSIS COMPLETE ===")
    print(f"Total correlations calculated: {len(results_df)}")
    print(f"Current correlations: {len(current_results)}")
    print(f"Forward lagged correlations: {len(forward_lagged)}")
    print(f"Reverse lagged correlations: {len(reverse_lagged)}")
    print(f"Significant correlations: {len(significant_results)}")
    print(f"Strong correlations (|r| >= 0.5): {len(strong_results)}")
    
    # Show top 10 strongest correlations
    print(f"\n=== TOP 10 STRONGEST CORRELATIONS ===")
    top_10 = results_df.head(10)
    for idx, row in top_10.iterrows():
        lag_desc = f"lag {row['Lag_Years']} years" if row['Lag_Years'] > 0 else f"reverse lag {abs(row['Lag_Years'])} years" if row['Lag_Years'] < 0 else "current"
        print(f"{row['Economic_Freedom_Indicator']} → {row['Outcome_Variable']} ({lag_desc}): r = {row['Correlation']:.4f}, n = {row['Sample_Size']}, p = {row['P_Value']:.4f}")
    
    print(f"\nResults saved to:")
    print(f"- data/results/comprehensive_correlation_analysis.csv")
    print(f"- data/results/comprehensive_correlation_analysis.xlsx")

if __name__ == "__main__":
    comprehensive_correlation_analysis() 