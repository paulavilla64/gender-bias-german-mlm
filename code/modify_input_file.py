import pandas as pd
import re
import chardet

df = pd.read_csv("../BEC-Pro/BEC-Pro_DE.tsv", sep='\t')

# modify the columns "Sent_AM" and "Sent_TAM"
# for any MASK except the first one, there should be BERUF or [PROF]

# Function to replace all [MASK]s with "Beruf" or [Prof] (for Sent_AM)
def replace_all_masks(sentence):
    return sentence.replace("[MASK]", "[PROF]")

# Function to replace all MASKs except the first one with "Beruf" or [PROF] (for Sent_TAM)
def replace_masks_except_first(sentence):
    # Find all [MASK] occurrences
    mask_positions = [m.start() for m in re.finditer(r"\[MASK\]", sentence)]
    
    # If there are more than one MASK, replace all but the first
    if len(mask_positions) > 1:
        first_mask_pos = mask_positions[0]
        modified_sentence = sentence[:first_mask_pos + 6]  # Keep the first [MASK]
        modified_sentence += sentence[first_mask_pos + 6:].replace("[MASK]", "[PROF]")
        return modified_sentence
    return sentence

# Apply the function to the relevant columns
#df["Sent_AM"] = df["Sent_AM"].apply(replace_all_masks)
#df["Sent_TAM"] = df["Sent_TAM"].apply(replace_masks_except_first)

# Save the modified CSV
#df.to_csv("../BEC-Pro/modified_file_DE_PROF.csv", sep='\t', index=False)

# then I also have to modify the padding with MASKS for maximum sentence length

def get_file_encoding(file_path):
    """Detect the encoding of a file."""
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
    return result['encoding']

def compare_text_files(file1, file2):
    """
    Compare two text files and check if they are completely identical.
    
    Returns:
        - True if files are the same.
        - False if they differ.
    """
    encoding1 = get_file_encoding(file1)
    encoding2 = get_file_encoding(file2)

    with open(file1, 'r', encoding=encoding1) as f1, open(file2, 'r', encoding=encoding2) as f2:
        for line1, line2 in zip(f1, f2):
            if line1 != line2:
                return False  # Files differ
        # Check if one file has extra lines
        if f1.read() or f2.read():
            return False
    return True  # Files are identical

# Example usage
file1 = "../data/results_with_modified_input_Beruf_DE_test_DE.csv"
file2 = "../data/results_with_modified_input_Beruf_PADs_DE.csv"
if compare_text_files(file1, file2):
   print("Files are identical.")
else:
   print("Files are different.")
