import pandas as pd
import re
import chardet
import json
import csv
import glob
import nltk
from nltk.tokenize import sent_tokenize

# Ensure you have the necessary NLTK resources
nltk.download('punkt')

df = pd.read_csv("../BEC-Pro/BEC-Pro_EN.tsv", sep='\t')

def save_unique_professions(tsv_path, output_txt_path):
    # Read the TSV file
    df = pd.read_csv(tsv_path, delimiter='\t', keep_default_na=False)  # Ensure NaNs don't cause issues
    
    # Convert 'Profession' column to a list while preserving order
    seen = set()
    unique_professions = [prof for prof in df['Profession'] if prof not in seen and not seen.add(prof)]
    
    # Write to a text file
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for profession in unique_professions:
            f.write(profession + '\n')

    print(f"Saved {len(unique_professions)} unique professions to {output_txt_path}")

# Example usage
# save_unique_professions("../BEC-Pro/BEC-Pro_EN.tsv", "../BEC-Pro/Professions/tokenized_professions_EN.txt")

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

# # Example usage
file1 = "../Gap/gap_flipped.tsv"
file2 = "../Gap/gap_flipped_adapted.tsv"
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


# input_file = "../BEC-Pro/Professions/tokenized_professions_EN.txt"  
# input_encoding = detect_encoding(input_file)
# output_file = "tokenized_professions_EN.txt" 
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
    df.loc[1620:1799, 'Sent_AM'] = [replace_mask(row, idx, professions_list_1_female, token_counts_1_female) for idx, row in df.loc[1620:1779].iterrows()]

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
# output_file = "output_tokenized_professions_DE.txt"  
# ts_file = "../BEC-Pro/BEC-Pro_DE.tsv"  
# updated_file = "updated_data_fixed_compounds.tsv"  

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
# df = pd.read_csv("../BEC-Pro/modified_file_EN_tokenized.tsv", sep='\t', encoding='utf-8')


# # Apply the function to modify Sent_TAM column
# df["Sent_TAM"] = df.apply(modify_sent_tam, axis=1)

# # Save the updated dataframe to a new file
# df.to_csv("updated_data_sent_TAM.tsv", sep='\t', index=False, encoding='utf-8')


def compare_profession_numbers(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Split into male and female professions
    male_professions = lines[:60]
    female_professions = lines[60:120]
    
    difference_count = 0
    
    for male_line, female_line in zip(male_professions, female_professions):
        male_number, male_profession = male_line.split(' ', 1)
        female_number, female_profession = female_line.split(' ', 1)
        
        male_number, female_number = int(male_number), int(female_number)
        difference = male_number - female_number
        
        if difference != 0:
            difference_count += 1
        
        print(f"{male_profession.strip()} vs {female_profession.strip()} -> Difference: {difference}")
    
    print(f"Total professions with a non-zero difference: {difference_count}")


# Example usage:
#file_path = "../BEC-Pro/tokenized_professions_DE.txt"  # Replace with actual file path
#compare_profession_numbers(file_path)

def replace_professions(csv_file, professions_file, output_file):
    encoding = detect_encoding(professions_file)

    # Read first 20 lines from the text file
    with open(professions_file, 'r', encoding=encoding, errors='replace') as f:
        professions = [line.split(':')[0].strip() for line in f.readlines()]

    # Divide professions into three lists
    professions_1 = professions[:20]    # First 20 for lines 0-1800
    professions_2 = professions[20:40]  # Next 20 for lines 1800-3600
    professions_3 = professions[40:60]  # Next 20 for lines 3600-5400

    # Load the CSV file
    df = pd.read_csv(csv_file, sep='\t', encoding='utf-8')

     # Replace professions in specified line ranges
    for i in range(1800):  # First 1800 rows
        df.at[i, 'Profession'] = professions_1[i % 20]  # Cycle through 20 professions

    for i in range(1800, 3600):  # Rows 1800-3600
        df.at[i, 'Profession'] = professions_2[(i - 1800) % 20]  # Cycle through professions_2

    for i in range(3600, 5400):  # Rows 3600-5400
        df.at[i, 'Profession'] = professions_3[(i - 3600) % 20]  # Cycle through professions_3


    # Save the modified CSV
    df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')

# Example usage
# csv_file = "../BEC-Pro/BEC-Pro_DE.tsv"  # Your actual input CSV file
# professions_file = "../BEC-Pro/professions_DE_gender_neutral.txt"  # Your actual text file
# output_file = "updated_data.csv"  # Output CSV file

# replace_professions(csv_file, professions_file, output_file)

def fill_templates(csv_file, output_file):
    # Load the CSV file
    df = pd.read_csv(csv_file, sep='\t', encoding='utf-8')

    # Function to replace placeholders with actual values
    def generate_sentence(row):
        template = row['Template']
        person = row['Person']
        profession = row['Profession']
        
        # Replace placeholders
        sentence = template.replace("<person subject>", person).replace("<profession>", profession)
        return sentence

    # Apply the function to update the 'Sentence' column
    df['Sentence'] = df.apply(generate_sentence, axis=1)

    # Save the modified CSV
    df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')

# Example usage
# csv_file = "updated_data.csv"  # Your input CSV file
# output_file = "final_data.csv"  # Output CSV file

#fill_templates(csv_file, output_file)


def modify_sentences(csv_file, output_file):
    # Load the CSV file
    df = pd.read_csv(csv_file, sep='\t', encoding='utf-8')

    # Define word mappings
    dieser_words = {"Mann"}
    diese_words = {"Frau"}
    mein_words = {"Bruder", "Sohn", "Ehemann", "Freund", "Vater", "Onkel", "Papa"}
    meine_words = {"Schwester", "Tochter", "Frau", "Freundin", "Mutter", "Tante", "Mama"}

    # Function to modify sentences
    def modify_sentence(sentence):
        words = sentence.split()

        # Iterate through words to modify them if needed
        for i, word in enumerate(words):
            if word in dieser_words:
                words[i] = "Dieser " + word
            elif word in diese_words:
                words[i] = "Diese " + word
            elif word in mein_words:
                words[i] = "Mein " + word
            elif word in meine_words:
                words[i] = "Meine " + word

        return " ".join(words)

    # Apply modifications to the 'Sentence' column
    df['Sentence'] = df['Sentence'].apply(modify_sentence)

    # Save the modified CSV
    df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')

def modify_sent_tm(csv_file, output_file):
    # Load the CSV file
    df = pd.read_csv(csv_file, sep='\t', encoding='utf-8')

    # Function to replace person words with [MASK]
    def mask_person(sentence, person_word):
        if pd.isna(person_word) or person_word.strip() == "":
            return sentence  # If no person word, keep the sentence unchanged
        return re.sub(r'\b' + re.escape(person_word) + r'\b', '[MASK]', sentence)

    # Modify the Sent_TM column
    df['Sent_TM'] = df.apply(lambda row: mask_person(row['Sentence'], row['Person']), axis=1)

    # Save the modified CSV
    df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')


# Example usage
# csv_file = "final_data_test.csv"  # Your input CSV file
# output_file = "final_data_test_1.csv"  # Output CSV file

#modify_sent_tm(csv_file, output_file)

def add_word_counts_to_professions(input_file, output_file):
    # Detect encoding
    encoding = detect_encoding(input_file)

    # Read the file
    with open(input_file, 'r', encoding=encoding, errors='replace') as f:
        lines = f.readlines()

    # Process each line and count words
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines:
            profession = line.strip()
            word_count = len(profession.split())  # Count words in profession name
            f.write(f"{word_count} {profession}\n")

# Example usage
# input_file = "../BEC-Pro/professions_DE_gender_neutral.txt"  # Replace with your input file
# output_file = "output_professions.txt"  # Replace with desired output file

# add_word_counts_to_professions(input_file, output_file)


def update_mask_count(csv_file, output_file):
    # Load the CSV file
    df = pd.read_csv(csv_file, sep='\t', encoding='utf-8')

    # Define professions with specific MASK token counts
    professions_2_mask = {"Dentalhygiene Fachkraft", "Naturheil Kraft", "Sachbearbeitende Person"}
    professions_3_mask = {
        "Fachkraft für Heizungstechnik", "Fachkraft für Kfz-Mechanik", "Einsatzkraft der Feuerwehr",
        "Fachkraft für Medizintechnik", "Fachkraft im Haarsalon", "Person am Empfang",
        "Fotografie betreibende Person", "medizinisch forschende Person", "Recht sprechende Person"
    }
    professions_4_mask = {
        "Mechanik Fachkraft für Busse", "Fachkraft für Holz-und Bautenschutzarbeiten",
        "Fachkraft in der Eisenbahn", "Servicekraft an der Bar"
    }
    professions_6_mask = {"Fachkraft für Kurier-, Express- und Postdienstleistungen"}

    all_target_professions = professions_2_mask | professions_3_mask | professions_4_mask | professions_6_mask

    # Function to modify Sent_AM based on profession
    def modify_sent_am(row):
        profession = row['Profession']

        # Determine the required MASK count
        if profession in professions_2_mask:
            mask_count = 2
        elif profession in professions_3_mask:
            mask_count = 3
        elif profession in professions_4_mask:
            mask_count = 4
        elif profession in professions_6_mask:
            mask_count = 6
        else:
            return row['Sent_AM']  # Keep unchanged if profession is not in the list

        # Step 1: Only replace multiple [MASK] tokens if the profession is in the list
        sentence_with_single_mask = row['Sent_AM']
        if profession in all_target_professions:
            sentence_with_single_mask = re.sub(r'\[MASK\]+', '[MASK]', row['Sent_AM'])

        # Step 2: Multiply the single [MASK] by the required count
        return sentence_with_single_mask.replace('[MASK]', ' '.join(['[MASK]'] * mask_count), 1)

    # Apply function to modify Sent_AM column
    df['Sent_AM'] = df.apply(modify_sent_am, axis=1)

    # Save the modified file
    df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')

# Example usage
# csv_file = "final_data_test_1.csv"  # Replace with actual input file
# output_file = "final_data_test_2.csv"  # Replace with desired output file

#update_mask_count(csv_file, output_file)

def update_sent_tam(csv_file, output_file):
    # Load the CSV file
    df = pd.read_csv(csv_file, sep='\t', encoding='utf-8')

    def modify_sent_tam(row):
        sent_tam = row['Sent_TAM']
        sent_am = row['Sent_AM']

        # Count the number of [MASK] tokens in Sent_AM
        mask_count_am = len(re.findall(r'\[MASK\]', sent_am))

        if mask_count_am < 1:
            return sent_tam  # No change needed if Sent_AM has no MASKs

        # Step 1: Ensure only the first TWO [MASK] remain
        mask_placeholders = ["[FIRST_MASK]", "[SECOND_MASK]"]  # Temporary markers
        new_sentence = sent_tam.replace("[MASK]", mask_placeholders[0], 1)  # Keep first MASK
        new_sentence = new_sentence.replace("[MASK]", mask_placeholders[1], 1)  # Keep second MASK
        new_sentence = re.sub(r'\[MASK\]', '', new_sentence).strip()  # Remove all other MASKs

        # Step 2: Expand the second [MASK] to the correct count
        new_sentence = new_sentence.replace(mask_placeholders[0], "[MASK]", 1)  # Restore first MASK
        new_sentence = new_sentence.replace(mask_placeholders[1], ' '.join(['[MASK]'] * mask_count_am), 1)  # Expand second MASK

        return new_sentence

    # Apply the function to the dataframe
    df['Sent_TAM'] = df.apply(modify_sent_tam, axis=1)

    # Save the modified dataframe
    df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')


# # Example usage
# csv_file = "modified_file_DE_gender_neutral_tokenized.tsv"  # Replace with actual input file
# output_file = "final_data_test_3.csv"  # Replace with desired output file

# update_sent_tam(csv_file, output_file)



def update_sent_am_EN(tsv_file, professions_file, output_file):
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
    # df.loc[:1799, 'Sent_AM'] = [replace_mask(row, idx, professions_list_1, token_counts_1) for idx, row in df.loc[:1799].iterrows()]

    # df.loc[1800:3599, 'Sent_AM'] = [replace_mask(row, idx, professions_list_2, token_counts_2) for idx, row in df.loc[1800:3599].iterrows()]
    
    # df.loc[3600:5399, 'Sent_AM'] = [replace_mask(row, idx, professions_list_3, token_counts_3) for idx, row in df.loc[3600:5399].iterrows()]

    
    # df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')

# Example usage:
# output_file = "../BEC-Pro/Professions/tokenized_professions_DE_gender_neutral.txt"  
# ts_file = "../BEC-Pro/modified_input/modified_file_DE_gender_neutral.csv"  
# updated_file = "../BEC-Pro/modified_file_DE_gender_neutral_tokenized.tsv"  

# update_sent_am_EN(ts_file, output_file, updated_file)


# Read CSV file
df = pd.read_csv("../BEC-Pro/BEC-Pro_DE.tsv", delimiter="\t")  

# Function to keep only the first [MASK]
def process_sent_tam(sent_tam):
    parts = sent_tam.split("[MASK]")
    if len(parts) > 1:
        return "[MASK]" + parts[1]  # Keep only the first MASK and the text after it
    return sent_tam  # Return unchanged if there's only one or no MASK

# Apply function to create new column
#df["Sent_TM_prior"] = df["Sent_TAM"].apply(process_sent_tam)

# df.loc[1080:1439, "Sent_TM_prior"] += ", hatte einen guten Arbeitstag."
# df.loc[2880:3239, "Sent_TM_prior"] += ", hatte einen guten Arbeitstag."
# df.loc[4680:5039, "Sent_TM_prior"] += ", hatte einen guten Arbeitstag."

# df.loc[720:1079, "Sent_TM_prior"] += "beworben."
# df.loc[2520:2879, "Sent_TM_prior"] += "beworben."
# df.loc[4320:4679, "Sent_TM_prior"] += "beworben."

# df.loc[1440:1799, "Sent_TM_prior"] += "werden."
# df.loc[3240:3559, "Sent_TM_prior"] += "werden."
# df.loc[5040:5399, "Sent_TM_prior"] += "werden."

# Save updated DataFrame
# df.to_csv("../BEC-Pro/modified_input/modified_file_DE_adapted_prior.csv", index=False, sep="\t")


def count_text_column_characters(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path, delimiter="\t")  # Adjust delimiter if needed

    # Ensure the "Text" column exists
    if "Text" not in df.columns:
        raise ValueError("The CSV file does not contain a 'Text' column.")

    # Count the characters in the "Text" column
    total_chars = df["Text"].astype(str).apply(len).sum()

    return total_chars

# Example usage
# file_path = "../data/gap_flipped.csv"  
# total_characters = count_text_column_characters(file_path)
# print(f"Total number of characters in the 'Text' column: {total_characters}")

def extract_text_and_id(input_file, output_file):
    # Load the CSV file
    df = pd.read_csv(input_file, delimiter="\t")  # Adjust delimiter if needed

    # Ensure the required columns exist
    required_columns = ["ID", "Text"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"The CSV file does not contain the '{col}' column.")

    # Select only the "ID" and "Text" columns
    df_filtered = df[["ID", "Text"]]

    # Save to a new CSV file
    df_filtered.to_csv(output_file, index=False, encoding="utf-8")

    print(f"Extracted data saved to {output_file}")

# Example usage
# input_file = "../data/gap_flipped.csv"   # Replace with your actual file path
# output_file = "../data/gap_flipped_structured.csv"   # Name of the new file
# extract_text_and_id(input_file, output_file)

def transfer_text_to_new_csv(input_csv, output_csv):
    """
    Reads text from the 'Text' column of the input CSV file and writes it into
    a new 'Text_German' column in the output CSV file.

    Args:
        input_csv (str): Path to the input CSV file (containing 'Text' column).
        output_csv (str): Path to the output CSV file (to which 'Text_German' is added).
    """
    try:
        # Load the input CSV and extract the 'Text' column
        df_input = pd.read_csv(input_csv)

        if "Text" not in df_input.columns:
            raise ValueError("Column 'Text' not found in the input CSV file.")

        extracted_text = df_input["Text"]

        # Load the output CSV
        df_output = pd.read_csv(output_csv, delimiter='\t')

        # Ensure lengths match, or trim/expand if needed
        min_length = min(len(df_output), len(extracted_text))
        df_output = df_output.iloc[:min_length]  # Trim output if it's longer
        extracted_text = extracted_text.iloc[:min_length]  # Trim input if longer

        # Add the new column
        df_output["Text_German"] = extracted_text.values

        # Save the updated CSV
        df_output.to_csv(output_csv, sep='\t', index=False)

        print(f"Updated CSV saved: {output_csv}")

    except Exception as e:
        print(f"Error: {e}")

# Example usage:
# transfer_text_to_new_csv("../Gap/gap_flipped_structured_translated.csv", "../Gap/gap_flipped.tsv")

def extract_fields_to_tsv(jsonl_files, output_tsv):
    """
    Extracts text from 'Neutral' and 'GenderStern' fields in multiple JSONL files,
    splits them by lines, and saves to a TSV file with an ID column.

    Parameters:
        jsonl_files (list): List of JSONL file paths.
        output_tsv (str): Path for the output TSV file.
    """
    with open(output_tsv, 'w', newline='', encoding='utf-8') as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        
        # Write header
        writer.writerow(["ID", "Neutral Text", "GenderStern Text"])

        line_id = 1  # ID counter

        for jsonl_file in jsonl_files:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())

                    # Extract and split 'Neutral' field
                    neutral_texts = data.get("Neutral", "").strip().split("\n")
                    neutral_texts = [text.strip() for text in neutral_texts if text.strip()]

                    # Extract and split 'GenderStern' field
                    genderstern_texts = data.get("GenderStern", "").strip().split("\n")
                    genderstern_texts = [text.strip() for text in genderstern_texts if text.strip()]

                    # Ensure we write rows even if one column is empty
                    max_lines = max(len(neutral_texts), len(genderstern_texts))
                    for i in range(max_lines):
                        neutral_line = neutral_texts[i] if i < len(neutral_texts) else ""
                        genderstern_line = genderstern_texts[i] if i < len(genderstern_texts) else ""
                        writer.writerow([line_id, neutral_line, genderstern_line])
                        line_id += 1  # Increment ID

    print(f"✅ Extracted 'Neutral' and 'GenderStern' fields from {len(jsonl_files)} files into {output_tsv} with IDs.")

# Example usage
#jsonl_files = glob.glob("../Lou/*.jsonl")  # Collect all JSONL files in the directory
#extract_fields_to_tsv(jsonl_files, "neutral_genderstern_texts.tsv")


def count_sentences_in_column(file_path, column_name):
    """
    Counts the total number of sentences in a specific column of a .tsv file.

    Parameters:
    - file_path: str, path to the .tsv file
    - column_name: str, name of the column to analyze

    Returns:
    - int, total count of sentences
    """
    # Read the TSV file
    df = pd.read_csv(file_path, delimiter='\t', dtype=str)  # Read as string to avoid NaNs

    # Check if the column exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the file.")

    # Count sentences
    total_sentences = sum(len(sent_tokenize(str(text))) for text in df[column_name].dropna())

    return total_sentences

# Example usage
#print(count_sentences_in_column("../Gap/gap_flipped.tsv", "Text"))


