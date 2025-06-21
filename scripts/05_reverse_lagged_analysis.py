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
        
        return correlation, p_value, ci_lower, ci_upper, se_corr
    else:
        return correlation, p_value, np.nan, np.nan, np.nan

def reverse_lagged_correlation_analysis(csv_filename='economic_freedom_gini_combined.csv'):
    """
    Analyze lagged correlations between Gini Index and economic freedom indicators.
    Tests if income inequality predicts future economic freedom.
    """
    print("Loading data...")
    df = pd.read_csv(csv_filename)
    
    # Sort by country and year to ensure proper lagging
    df = df.sort_values(['Country', 'Index Year'])
    
    # Define the economic freedom indicators
    indicators = [
        'Overall Score', 'Property Rights', 'Government Integrity', 'Judicial Effectiveness',
        'Tax Burden', 'Government Spending', 'Fiscal Health', 'Business Freedom',
        'Labor Freedom', 'Monetary Freedom', 'Trade Freedom', 'Investment Freedom',
        'Financial Freedom'
    ]
    
    print(f"Analyzing reverse lagged correlations for {len(indicators)} indicators...")
    print(f"Testing if Gini Index predicts future economic freedom...")
    print(f"Testing lags: 1-10 years, then 15, 20, 25, 30 years...")
    
    all_results = []
    
    for indicator in indicators:
        if indicator in df.columns:
            # Create lag periods: 1-10, then 15, 20, 25, 30
            lag_periods = list(range(1, 11)) + [15, 20, 25, 30]
            
            for lag in lag_periods:
                # Create lagged version of Gini Index (Gini Index lagged to predict future economic freedom)
                df_lagged = df.copy()
                df_lagged[f'Gini_Index_lag_{lag}'] = df_lagged.groupby('Country')['Gini_Index'].shift(lag)
                
                # Clean data - remove rows with missing values
                clean_df = df_lagged.dropna(subset=[indicator, f'Gini_Index_lag_{lag}'])
                
                if len(clean_df) > 0:
                    # Calculate correlation and statistics
                    correlation, p_value, ci_lower, ci_upper, se_corr = calculate_correlation_stats(
                        clean_df[f'Gini_Index_lag_{lag}'], clean_df[indicator]
                    )
                    
                    # Calculate additional statistics
                    sample_size = len(clean_df)
                    gini_mean = clean_df[f'Gini_Index_lag_{lag}'].mean()
                    indicator_mean = clean_df[indicator].mean()
                    
                    # Only include results with sample size >= 50
                    if sample_size >= 50:
                        all_results.append({
                            'Indicator': indicator,
                            'Lag_Years': lag,
                            'Correlation_with_Gini': round(correlation, 4),
                            'P_Value': round(p_value, 4),
                            'Standard_Error': round(se_corr, 4),
                            'CI_Lower_95': round(ci_lower, 4),
                            'CI_Upper_95': round(ci_upper, 4),
                            'Sample_Size': sample_size,
                            'Gini_Mean': round(gini_mean, 4),
                            'Indicator_Mean': round(indicator_mean, 4),
                            'Significant_05': p_value < 0.05,
                            'Significant_01': p_value < 0.01
                        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) == 0:
        print("\nNo significant correlations found.")
        return results_df
    
    # Sort by absolute correlation value
    results_df = results_df.sort_values('Correlation_with_Gini', key=abs, ascending=False)
    
    # Save results
    output_filename = 'reverse_lagged_correlation_analysis.csv'
    excel_filename = 'reverse_lagged_correlation_analysis.xlsx'
    
    print(f"\nSaving results to: {output_filename}")
    results_df.to_csv(output_filename, index=False)
    
    # Try to save Excel file, but don't fail if there's a permission error
    try:
        results_df.to_excel(excel_filename, index=False)
        print(f"Also saved to: {excel_filename}")
    except PermissionError:
        print(f"Could not save Excel file (may be open in another program): {excel_filename}")
    except Exception as e:
        print(f"Could not save Excel file: {e}")
    
    # Print summary statistics
    print(f"\n{'='*100}")
    print(f"REVERSE LAGGED CORRELATION ANALYSIS SUMMARY")
    print(f"{'='*100}")
    print(f"Total correlations calculated: {len(results_df)}")
    print(f"Statistically significant (p < 0.05): {len(results_df[results_df['P_Value'] < 0.05])}")
    print(f"Highly significant (p < 0.001): {len(results_df[results_df['P_Value'] < 0.001])}")
    
    # Show top correlations
    print(f"\nTop 10 Strongest Correlations (Gini Index → Economic Freedom):")
    print(f"{'='*80}")
    top_10 = results_df.head(10)
    for _, row in top_10.iterrows():
        direction = "positive" if row['Correlation_with_Gini'] > 0 else "negative"
        print(f"{row['Indicator']} (lag {row['Lag_Years']} years): {row['Correlation_with_Gini']:.4f} ({direction}, n={row['Sample_Size']}, p={row['P_Value']:.4f})")
    
    # Show best lag for each indicator
    print(f"\nBest Lag Period for Each Indicator (Significant Only):")
    print(f"{'='*80}")
    significant_results = results_df[results_df['P_Value'] < 0.05]
    if len(significant_results) > 0:
        best_lags = significant_results.loc[significant_results.groupby('Indicator')['Correlation_with_Gini'].idxmax()]
        for _, row in best_lags.iterrows():
            direction = "positive" if row['Correlation_with_Gini'] > 0 else "negative"
            print(f"{row['Indicator']}: {row['Correlation_with_Gini']:.4f} at {row['Lag_Years']} years lag ({direction}, p={row['P_Value']:.4f})")
    
    return results_df

if __name__ == "__main__":
    results = reverse_lagged_correlation_analysis() 