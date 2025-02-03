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
file1 = "../data/modified_input/results_DE_with_padding_DE.csv"
file2 = "../data/results_with_tokens_DE.csv"
if compare_text_files(file1, file2):
   print("Files are identical.")
else:
   print("Files are different.")


def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']


def add_token_counts_to_file(input_file, output_file):
    with open(input_file, 'r', encoding=input_encoding, errors='replace') as f:
        lines = f.readlines()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines:
            if not line.strip():
                continue  # Skip empty lines
            
            profession, tokens = line.split(':')
            tokens = eval(tokens.strip())  # Convert string list to actual list
            token_count = len(tokens)
            
            f.write(f"{token_count} {profession}: {tokens}\n")


input_file = "tokenized_professions_DE.txt"  
#input_encoding = detect_encoding(input_file)
output_file = "output_professions.txt" 
# add_token_counts_to_file(input_file, output_file)

def update_sent_am(tsv_file, professions_file, output_file):
    df = pd.read_csv(tsv_file, sep='\t', encoding='utf-8')
    
    with open(professions_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    token_counts_1 = {}
    token_counts_2 = {}
    token_counts_3 = {}
    token_counts_1_female = {}
    token_counts_2_female = {}
    token_counts_3_female = {}
    professions_list_1 = []
    professions_list_2 = []
    professions_list_3 = []
    professions_list_1_female = []
    professions_list_2_female = []
    professions_list_3_female= []

    for line in lines[:20]:  # Only process the first 20 lines
        parts = line.split(' ', 1)
        if len(parts) < 2:
            continue
        count, profession = parts
        count = int(count)
        profession = profession.split(':')[0].strip()
        token_counts_1[profession] = count
        professions_list_1.append(profession)
    
    for line in lines[20:40]:  # Next 20 professions
        parts = line.split(' ', 1)
        if len(parts) < 2:
            continue
        count, profession = parts
        count = int(count)
        profession = profession.split(':')[0].strip()
        token_counts_2[profession] = count
        professions_list_2.append(profession)
    
    for line in lines[40:60]:  # Next 20 professions
        parts = line.split(' ', 1)
        if len(parts) < 2:
            continue
        count, profession = parts
        count = int(count)
        profession = profession.split(':')[0].strip()
        token_counts_3[profession] = count
        professions_list_3.append(profession)
    
    for line in lines[60:80]:  # Next 20 professions
        parts = line.split(' ', 1)
        if len(parts) < 2:
            continue
        count, profession = parts
        count = int(count)
        profession = profession.split(':')[0].strip()
        token_counts_1_female[profession] = count
        professions_list_1_female.append(profession)
    
    for line in lines[80:100]:  # Next 20 professions
        parts = line.split(' ', 1)
        if len(parts) < 2:
            continue
        count, profession = parts
        count = int(count)
        profession = profession.split(':')[0].strip()
        token_counts_2_female[profession] = count
        professions_list_2_female.append(profession)
    
    for line in lines[100:120]:  # Next 20 professions
        parts = line.split(' ', 1)
        if len(parts) < 2:
            continue
        count, profession = parts
        count = int(count)
        profession = profession.split(':')[0].strip()
        token_counts_3_female[profession] = count
        professions_list_3_female.append(profession)
    
    # fix: for compounds, it should not just replace one MASK, but all MASKs
    def replace_mask(row, index, professions_list, token_counts):
        profession_index = index % 20  # Cycle through the first 20 professions
        profession = professions_list[profession_index]
        token_count = token_counts.get(profession, 1)  # Default to 1 if not found

        # Step 1: Ensure only ONE [MASK] remains
        mask_placeholder = "[TEMP_MASK]"  # Temporary marker to preserve first occurrence
        new_sentence = row['Sent_AM'].replace("[MASK]", mask_placeholder, 1)  # Keep first MASK
        new_sentence = re.sub(r'\[MASK\]', '', new_sentence).strip()  # Remove all other MASKs

        # Step 2: Expand the preserved [MASK] to the correct count
        new_sentence = new_sentence.replace(mask_placeholder, ' '.join(['[MASK]'] * token_count), 1)

        return new_sentence
    
    # male professions for templates
    df.loc[:179, 'Sent_AM'] = [replace_mask(row, idx, professions_list_1, token_counts_1) for idx, row in df.loc[:179].iterrows()]
    df.loc[360:539, 'Sent_AM'] = [replace_mask(row, idx, professions_list_1, token_counts_1) for idx, row in df.loc[360:539].iterrows()]
    df.loc[720:899, 'Sent_AM'] = [replace_mask(row, idx, professions_list_1, token_counts_1) for idx, row in df.loc[720:899].iterrows()]
    df.loc[1080:1259, 'Sent_AM'] = [replace_mask(row, idx, professions_list_1, token_counts_1) for idx, row in df.loc[1080:1259].iterrows()]
    df.loc[1440:1619, 'Sent_AM'] = [replace_mask(row, idx, professions_list_1, token_counts_1) for idx, row in df.loc[1440:1619].iterrows()]

    df.loc[1800:1979, 'Sent_AM'] = [replace_mask(row, idx, professions_list_2, token_counts_2) for idx, row in df.loc[1800:1979].iterrows()]
    df.loc[2160:2339, 'Sent_AM'] = [replace_mask(row, idx, professions_list_2, token_counts_2) for idx, row in df.loc[2160:2339].iterrows()]
    df.loc[2520:2699, 'Sent_AM'] = [replace_mask(row, idx, professions_list_2, token_counts_2) for idx, row in df.loc[2520:2699].iterrows()]
    df.loc[2880:3059, 'Sent_AM'] = [replace_mask(row, idx, professions_list_2, token_counts_2) for idx, row in df.loc[2880:3059].iterrows()]
    df.loc[3240:3419, 'Sent_AM'] = [replace_mask(row, idx, professions_list_2, token_counts_2) for idx, row in df.loc[3240:3419].iterrows()]

    df.loc[3600:3779, 'Sent_AM'] = [replace_mask(row, idx, professions_list_3, token_counts_3) for idx, row in df.loc[3600:3779].iterrows()]
    df.loc[3960:4139, 'Sent_AM'] = [replace_mask(row, idx, professions_list_3, token_counts_3) for idx, row in df.loc[3960:4139].iterrows()]
    df.loc[4320:4499, 'Sent_AM'] = [replace_mask(row, idx, professions_list_3, token_counts_3) for idx, row in df.loc[4320:4499].iterrows()]
    df.loc[4680:4859, 'Sent_AM'] = [replace_mask(row, idx, professions_list_3, token_counts_3) for idx, row in df.loc[4680:4859].iterrows()]
    df.loc[5040:5219, 'Sent_AM'] = [replace_mask(row, idx, professions_list_3, token_counts_3) for idx, row in df.loc[5040:5219].iterrows()]

    # female professions for templates
    df.loc[180:359, 'Sent_AM'] = [replace_mask(row, idx, professions_list_1_female, token_counts_1_female) for idx, row in df.loc[180:359].iterrows()]
    df.loc[540:719, 'Sent_AM'] = [replace_mask(row, idx, professions_list_1_female, token_counts_1_female) for idx, row in df.loc[540:719].iterrows()]
    df.loc[900:1079, 'Sent_AM'] = [replace_mask(row, idx, professions_list_1_female, token_counts_1_female) for idx, row in df.loc[900:1079].iterrows()]
    df.loc[1260:1439, 'Sent_AM'] = [replace_mask(row, idx, professions_list_1_female, token_counts_1_female) for idx, row in df.loc[1260:1439].iterrows()]
    df.loc[1620:1779, 'Sent_AM'] = [replace_mask(row, idx, professions_list_1_female, token_counts_1_female) for idx, row in df.loc[1620:1779].iterrows()]

    df.loc[1980:2159, 'Sent_AM'] = [replace_mask(row, idx, professions_list_2_female, token_counts_2_female) for idx, row in df.loc[1980:2159].iterrows()]
    df.loc[2340:2519, 'Sent_AM'] = [replace_mask(row, idx, professions_list_2_female, token_counts_2_female) for idx, row in df.loc[2340:2519].iterrows()]
    df.loc[2700:2879, 'Sent_AM'] = [replace_mask(row, idx, professions_list_2_female, token_counts_2_female) for idx, row in df.loc[2700:2879].iterrows()]
    df.loc[3060:3239, 'Sent_AM'] = [replace_mask(row, idx, professions_list_2_female, token_counts_2_female) for idx, row in df.loc[3060:3239].iterrows()]
    df.loc[3420:3599, 'Sent_AM'] = [replace_mask(row, idx, professions_list_2_female, token_counts_2_female) for idx, row in df.loc[3420:3599].iterrows()]

    df.loc[3780:3959, 'Sent_AM'] = [replace_mask(row, idx, professions_list_3_female, token_counts_3_female) for idx, row in df.loc[3780:3959].iterrows()]
    df.loc[4140:4319, 'Sent_AM'] = [replace_mask(row, idx, professions_list_3_female, token_counts_3_female) for idx, row in df.loc[4140:4319].iterrows()]
    df.loc[4500:4679, 'Sent_AM'] = [replace_mask(row, idx, professions_list_3_female, token_counts_3_female) for idx, row in df.loc[4500:4679].iterrows()]
    df.loc[4860:5039, 'Sent_AM'] = [replace_mask(row, idx, professions_list_3_female, token_counts_3_female) for idx, row in df.loc[4860:5039].iterrows()]
    df.loc[5220:5399, 'Sent_AM'] = [replace_mask(row, idx, professions_list_3_female, token_counts_3_female) for idx, row in df.loc[5220:5399].iterrows()]
    
    df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')

# Example usage:
output_file = "output_tokenized_professions_DE.txt"  
ts_file = "../BEC-Pro/BEC-Pro_DE.tsv"  
updated_file = "updated_data_fixed_compounds.tsv"  

#update_sent_am(ts_file, output_file, updated_file)

# now adapt sent_TAM
# all MASKs, starting from the second MASK, should be replaced with the amount of MASKs from sent_am


def modify_sent_tam(row):
    # Get the count of [MASK] tokens in Sent_AM
    sent_am_mask_count = row["Sent_AM"].count("[MASK]")
    
    # Step 1: Preserve the first [MASK] and replace the second one with a temporary placeholder
    mask_placeholder = "[TEMP_MASK]"
    new_sentence = row["Sent_TAM"]
    
    # Find the positions of all [MASK] occurrences in the sentence
    mask_positions = [m.start() for m in re.finditer(r'\[MASK\]', new_sentence)]
    
    if len(mask_positions) > 1:
        # Replace the second [MASK] with the placeholder, leaving the first [MASK] untouched
        second_mask_position = mask_positions[1]
        new_sentence = new_sentence[:second_mask_position] + mask_placeholder + new_sentence[second_mask_position+6:]
    
    # Step 2: Remove all other [MASK] occurrences (except the first one)
    # Now we only remove [MASK] tokens that come after the first one
    mask_count = 0
    def remove_extra_masks(match):
        nonlocal mask_count
        mask_count += 1
        return '' if mask_count > 1 else match.group(0)

    new_sentence = re.sub(r'\[MASK\]', remove_extra_masks, new_sentence)
    
    # Step 3: Replace the placeholder with the correct number of [MASK] tokens
    new_sentence = new_sentence.replace(mask_placeholder, ' '.join(['[MASK]'] * sent_am_mask_count), 1)
    
    return new_sentence

# Load the TSV file
#df = pd.read_csv("updated_data_fixed_compounds.tsv", sep='\t', encoding='utf-8')


# Apply the function to modify Sent_TAM column
#df["Sent_TAM"] = df.apply(modify_sent_tam, axis=1)

# Save the updated dataframe to a new file
#df.to_csv("updated_data_sent_TAM_fixed.tsv", sep='\t', index=False, encoding='utf-8')

