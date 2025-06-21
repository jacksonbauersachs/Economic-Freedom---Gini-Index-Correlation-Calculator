import pandas as pd
import numpy as np

def comprehensive_correlation_analysis(csv_filename='economic_freedom_gini_combined.csv', output_filename='correlation_analysis_results.csv'):
    """
    Calculate correlations between Gini Index and all economic freedom scores.
    Output results to a CSV file with labels and sample sizes.
    """
    print("Loading data...")
    df = pd.read_csv(csv_filename)
    
    # Define the economic freedom indicators and their descriptions
    indicators = {
        'Overall Score': 'Overall Economic Freedom Score (0-100)',
        'Property Rights': 'Property Rights Score (0-100)',
        'Government Integrity': 'Government Integrity Score (0-100)',
        'Judicial Effectiveness': 'Judicial Effectiveness Score (0-100)',
        'Tax Burden': 'Tax Burden Score (0-100)',
        'Government Spending': 'Government Spending Score (0-100)',
        'Fiscal Health': 'Fiscal Health Score (0-100)',
        'Business Freedom': 'Business Freedom Score (0-100)',
        'Labor Freedom': 'Labor Freedom Score (0-100)',
        'Monetary Freedom': 'Monetary Freedom Score (0-100)',
        'Trade Freedom': 'Trade Freedom Score (0-100)',
        'Investment Freedom': 'Investment Freedom Score (0-100)',
        'Financial Freedom': 'Financial Freedom Score (0-100)'
    }
    
    print("Calculating correlations...")
    results = []
    
    for indicator, description in indicators.items():
        if indicator in df.columns:
            # Drop rows with missing values for this specific indicator and Gini_Index
            clean_df = df.dropna(subset=[indicator, 'Gini_Index'])
            
            if len(clean_df) > 0:
                # Calculate correlation
                correlation = clean_df[indicator].corr(clean_df['Gini_Index'])
                
                # Calculate additional statistics
                sample_size = len(clean_df)
                gini_mean = clean_df['Gini_Index'].mean()
                gini_std = clean_df['Gini_Index'].std()
                indicator_mean = clean_df[indicator].mean()
                indicator_std = clean_df[indicator].std()
                
                results.append({
                    'Indicator': indicator,
                    'Description': description,
                    'Correlation_with_Gini': correlation,
                    'Sample_Size': sample_size,
                    'Gini_Mean': gini_mean,
                    'Gini_Std': gini_std,
                    'Indicator_Mean': indicator_mean,
                    'Indicator_Std': indicator_std
                })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by absolute correlation value (strongest correlations first)
    results_df['Abs_Correlation'] = abs(results_df['Correlation_with_Gini'])
    results_df = results_df.sort_values('Abs_Correlation', ascending=False)
    results_df = results_df.drop('Abs_Correlation', axis=1)
    
    # Round numeric columns for better readability
    numeric_columns = ['Correlation_with_Gini', 'Gini_Mean', 'Gini_Std', 'Indicator_Mean', 'Indicator_Std']
    for col in numeric_columns:
        if col in results_df.columns:
            results_df[col] = results_df[col].round(4)
    
    # Save to CSV
    print(f"Saving results to: {output_filename}")
    results_df.to_csv(output_filename, index=False)
    
    # Also save to Excel for better formatting
    excel_filename = output_filename.replace('.csv', '.xlsx')
    results_df.to_excel(excel_filename, index=False)
    
    # Print summary
    print(f"\nCorrelation Analysis Complete!")
    print(f"Results saved to: {output_filename} and {excel_filename}")
    print(f"Total indicators analyzed: {len(results_df)}")
    
    # Print top correlations
    print(f"\nTop 5 Strongest Correlations with Gini Index:")
    print("=" * 80)
    for i, row in results_df.head().iterrows():
        print(f"{row['Indicator']}: {row['Correlation_with_Gini']:.4f} (n={row['Sample_Size']})")
        print(f"  Description: {row['Description']}")
        print()
    
    return results_df

if __name__ == "__main__":
    results = comprehensive_correlation_analysis() 