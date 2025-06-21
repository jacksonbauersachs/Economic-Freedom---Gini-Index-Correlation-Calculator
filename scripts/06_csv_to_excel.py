import pandas as pd

def csv_to_excel(csv_filename='economic_freedom_gini_combined.csv', excel_filename='economic_freedom_gini_combined.xlsx'):
    """
    Convert the cleaned CSV file to Excel format.
    """
    print(f"Loading CSV file: {csv_filename}")
    df = pd.read_csv(csv_filename)
    
    print(f"Converting to Excel: {excel_filename}")
    df.to_excel(excel_filename, index=False)
    
    print(f"Successfully exported to Excel!")
    print(f"File saved as: {excel_filename}")
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    csv_to_excel() 