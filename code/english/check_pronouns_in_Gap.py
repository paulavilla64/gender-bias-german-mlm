import pandas as pd

def extract_unique_pronouns(tsv_filepath):
    """
    Extract all unique pronouns from the 'Pronoun' column of a TSV file
    
    Args:
        tsv_filepath (str): Path to the TSV file
        
    Returns:
        list: Sorted list of unique pronouns
    """
    
    try:
        # Read the TSV file
        df = pd.read_csv(tsv_filepath, sep='\t')
        
        # Check if 'Pronoun' column exists
        if 'Pronoun' not in df.columns:
            print("Error: 'Pronoun' column not found in the TSV file")
            return []
        
        # Extract unique pronouns and convert to list
        unique_pronouns = sorted(df['Pronoun'].unique().tolist())
        
        # Print summary
        print(f"Found {len(unique_pronouns)} unique pronouns in the file")
        
        return unique_pronouns
    
    except Exception as e:
        print(f"Error reading TSV file: {e}")
        return []

# Example usage
if __name__ == "__main__":
    filepath = "../../datasets/Gap/gap_flipped.tsv" 
    pronouns = extract_unique_pronouns(filepath)
    print("Unique pronouns found:")
    for pronoun in pronouns:
        print(f"- {pronoun}")