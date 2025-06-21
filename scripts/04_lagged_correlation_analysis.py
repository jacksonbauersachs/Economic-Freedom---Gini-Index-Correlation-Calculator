import pandas as pd
import numpy as np
from scipy import stats

def calculate_correlation_stats(x, y):
    """
    Calculate correlation with confidence interval and p-value.
    """
    # Calculate correlation
    correlation, p_value = stats.pearsonr(x, y)
    
    # Calculate confidence interval using Fisher's z-transformation
    n = len(x)
    if n > 3:
        # Fisher's z-transformation
        z = 0.5 * np.log((1 + correlation) / (1 - correlation))
        
        # Standard error of z
        se_z = 1 / np.sqrt(n - 3)
        
        # 95% confidence interval for z
        z_lower = z - 1.96 * se_z
        z_upper = z + 1.96 * se_z
        
        # Transform back to correlation scale
        ci_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        ci_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        
        # Standard error of correlation
        se_corr = np.sqrt((1 - correlation**2) / (n - 2))
    else:
        ci_lower = ci_upper = se_corr = np.nan
    
    return correlation, p_value, ci_lower, ci_upper, se_corr

def lagged_correlation_analysis(csv_filename='economic_freedom_gini_combined.csv', max_lag=30):
    """
    Analyze lagged correlations between economic freedom indicators and Gini index.
    Tests lags from 1 to max_lag years to see if economic freedom predicts future inequality.
    """
    print("Loading data...")
    df = pd.read_csv(csv_filename)
    
    # Sort by country and year to ensure proper lagging
    df = df.sort_values(['Country', 'Index Year'])
    
    # Define the economic freedom indicators
    indicators = [
        'Overall Score', 'Property Rights', 'Government Integrity', 'Judicial Effectiveness',
        'Tax Burden', 'Government Spending', 'Fiscal Health', 'Business Freedom',
        'Labor Freedom', 'Monetary Freedom', 'Trade Freedom', 'Investment Freedom', 'Financial Freedom'
    ]
    
    print(f"Analyzing lagged correlations for {len(indicators)} indicators...")
    print(f"Testing lags: 1-10 years, then 15, 20, 25, 30 years...")
    
    all_results = []
    
    for indicator in indicators:
        if indicator in df.columns:
            print(f"\nAnalyzing: {indicator}")
            
            # Create lag periods: 1-10, then 15, 20, 25, 30
            lag_periods = list(range(1, 11)) + [15, 20, 25, 30]
            
            for lag in lag_periods:
                # Create lagged version of the indicator
                df_lagged = df.copy()
                df_lagged[f'{indicator}_lag_{lag}'] = df_lagged.groupby('Country')[indicator].shift(lag)
                
                # Drop rows where we don't have both current Gini and lagged indicator
                clean_df = df_lagged.dropna(subset=['Gini_Index', f'{indicator}_lag_{lag}'])
                
                if len(clean_df) > 0:
                    # Calculate correlation with confidence intervals and p-value
                    correlation, p_value, ci_lower, ci_upper, se_corr = calculate_correlation_stats(
                        clean_df['Gini_Index'], clean_df[f'{indicator}_lag_{lag}']
                    )
                    
                    # Calculate additional statistics
                    sample_size = len(clean_df)
                    gini_mean = clean_df['Gini_Index'].mean()
                    indicator_mean = clean_df[f'{indicator}_lag_{lag}'].mean()
                    
                    # Only include results with sample size >= 50
                    if sample_size >= 50:
                        all_results.append({
                            'Indicator': indicator,
                            'Lag_Years': lag,
                            'Correlation_with_Gini': correlation,
                            'P_Value': p_value,
                            'Standard_Error': se_corr,
                            'CI_Lower_95': ci_lower,
                            'CI_Upper_95': ci_upper,
                            'Sample_Size': sample_size,
                            'Gini_Mean': gini_mean,
                            'Indicator_Mean': indicator_mean,
                            'Significant_05': p_value < 0.05,
                            'Significant_01': p_value < 0.01
                        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Round numeric columns
    numeric_columns = ['Correlation_with_Gini', 'P_Value', 'Standard_Error', 'CI_Lower_95', 'CI_Upper_95', 'Gini_Mean', 'Indicator_Mean']
    for col in numeric_columns:
        if col in results_df.columns:
            results_df[col] = results_df[col].round(4)
    
    # Sort by absolute correlation value
    results_df['Abs_Correlation'] = abs(results_df['Correlation_with_Gini'])
    results_df = results_df.sort_values('Abs_Correlation', ascending=False)
    results_df = results_df.drop('Abs_Correlation', axis=1)
    
    # Save results
    output_filename = 'lagged_correlation_analysis.csv'
    excel_filename = 'lagged_correlation_analysis.xlsx'
    
    print(f"\nSaving results to: {output_filename} and {excel_filename}")
    results_df.to_csv(output_filename, index=False)
    results_df.to_excel(excel_filename, index=False)
    
    # Print summary statistics
    print(f"\nLagged Correlation Analysis Complete!")
    print(f"Total correlations calculated: {len(results_df)}")
    
    # Count significant correlations
    significant_05 = results_df['Significant_05'].sum()
    significant_01 = results_df['Significant_01'].sum()
    print(f"Statistically significant correlations (p < 0.05): {significant_05}")
    print(f"Highly significant correlations (p < 0.01): {significant_01}")
    
    # Show top correlations by lag
    print(f"\nTop 10 Strongest Lagged Correlations:")
    print("=" * 100)
    for i, row in results_df.head(10).iterrows():
        direction = "positive" if row['Correlation_with_Gini'] > 0 else "negative"
        significance = "***" if row['P_Value'] < 0.01 else "**" if row['P_Value'] < 0.05 else "*" if row['P_Value'] < 0.1 else ""
        print(f"{row['Indicator']} (lag {row['Lag_Years']} years): {row['Correlation_with_Gini']:.4f} ({direction}, n={row['Sample_Size']}, p={row['P_Value']:.4f}) {significance}")
        print(f"  95% CI: [{row['CI_Lower_95']:.4f}, {row['CI_Upper_95']:.4f}], SE: {row['Standard_Error']:.4f}")
    
    # Show best lag for each indicator
    print(f"\nBest Lag Period for Each Indicator (Significant Only):")
    print("=" * 100)
    for indicator in indicators:
        if indicator in results_df['Indicator'].values:
            indicator_results = results_df[results_df['Indicator'] == indicator]
            # Filter for significant correlations only
            significant_results = indicator_results[indicator_results['P_Value'] < 0.05]
            if len(significant_results) > 0:
                best_result = significant_results.iloc[0]  # Already sorted by absolute correlation
                direction = "positive" if best_result['Correlation_with_Gini'] > 0 else "negative"
                print(f"{indicator}: {best_result['Correlation_with_Gini']:.4f} at {best_result['Lag_Years']} years lag ({direction}, p={best_result['P_Value']:.4f})")
    
    return results_df

if __name__ == "__main__":
    results = lagged_correlation_analysis() 