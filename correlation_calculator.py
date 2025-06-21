import pandas as pd

def calculate_correlation(filename='economic_freedom_gini_combined.csv'):
    """
    Calculate and print the Pearson correlation between Overall Score and Gini_Index.
    """
    df = pd.read_csv(filename)
    # Drop rows with missing values in either column
    df = df.dropna(subset=['Overall Score', 'Gini_Index'])
    corr = df['Overall Score'].corr(df['Gini_Index'])
    print(f"Pearson correlation between Overall Score and Gini Index: {corr:.4f}")
    return corr

if __name__ == "__main__":
    calculate_correlation() 