from datasets import load_dataset
import datasets
import random
import re

# print("Downloading AG News dataset...")
# dataset = load_dataset("ag_news", split="train", cache_dir="./data")

# print("✅ Dataset downloaded successfully!")
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import csv

# # Attempt to read using IPC
# with pa.memory_map("./data/ag_news/default/0.0.0/eb185aade064a813bc0b7f42de02595523103ca4/ag_news-train.arrow", "rb") as source:
#     reader = ipc.open_file(source)
#     table = reader.read_all()

# # Convert to Pandas for easy exploration
# df = table.to_pandas()

# # Print first rows
# print(df.head())

# with open("./data/ag_news/default/0.0.0/eb185aade064a813bc0b7f42de02595523103ca4/ag_news-train.arrow", "rb") as f:
#     print(f.read(16))  # Read first 16 bytes

# Force redownload
# dataset = load_dataset("ag_news", split="train", download_mode="force_redownload")

# Print some example data
# print(dataset)
# print(dataset[0])  # Print the first record

import pandas as pd

# df = pd.DataFrame(dataset)
# print(df.head())

import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

# # Extract sentences from the first 5 news articles
# for i in range(5):
#     sentences = sent_tokenize(dataset[i]["text"])
#     cleaned_sentences = [s.replace("\\", "") for s in sentences]

#     print(f"Article {i+1}: {cleaned_sentences}\n")

def extract_and_save_sentences(input_file, num_sentences=268726, output_file="ag_news_common_crawl_expanded.tsv"):
    """
    Reads a dataset from an .arrow file, extracts a specified number of sentences, and saves them to a .tsv file.
    
    Parameters:
        input_file (str): Path to the .arrow dataset file.
        num_sentences (int): Number of sentences to extract.
        output_file (str): Name of the output .tsv file.
    """
    # Load the dataset from the .arrow file
    dataset = datasets.Dataset.from_file(input_file)

    sentences = []
    
    # Extract sentences from dataset
    for row in dataset:
        text = row["text"]  # Adjust this if your column name is different
        sentences.extend(sent_tokenize(text))  # Tokenize into sentences
        
        if len(sentences) >= num_sentences:
            break

    sentences = sentences[:num_sentences]  # Ensure we only take the required number

    # Save to .tsv file
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["Sentence"])  # Header
        for sentence in sentences:
            writer.writerow([sentence])

    print(f"Saved {num_sentences} sentences to {output_file}")

# Example usage:
input_file = "../ag_news/data/ag_news/default/0.0.0/eb185aade064a813bc0b7f42de02595523103ca4/ag_news-train.arrow"

#extract_and_save_sentences(input_file)

def extract_sentences_to_tsv(input_file, output_tsv, num_sentences):
    """
    Reads a .txt file, extracts a specified number of sentences, and saves them to a .tsv file.
    
    Args:
        input_txt (str): Path to the input text file.
        output_tsv (str): Path to the output TSV file.
        num_sentences (int): Number of sentences to extract.
    """
    sentences = []
    # Read the input text file
    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
                parts = line.strip().split("\t", 1)  # Split into ID and sentence (if tab-separated)
                if len(parts) == 2:
                    sentence = parts[1]  # Keep only the sentence text
                else:
                    sentence = parts[0]  # If no tab, assume it's just a sentence
                
                sentences.append(sentence)

                if len(sentences) >= num_sentences:
                    break  # Stop when we reach the required number


    # Write to a TSV file
    with open(output_tsv, 'w', encoding='utf-8', newline="") as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Sentence"])  # Column header
        for sentence in sentences:
            writer.writerow([sentence])

    print(f"Extracted {len(sentences)} sentences and saved them to {output_tsv}")

# Example usage
# extract_sentences_to_tsv("../Leipzig_corpus/deu_news_2024_30K-sentences.txt", "deu_news_leipzig.tsv", 13436)

def extract_person_words_by_gender(csv_file, output_file):
    """
    Extract unique person words from a CSV file, separated by gender.
    
    Args:
        csv_file (str): Path to the CSV file
        output_file (str): Path to save the output text file
    """
    # Read the CSV file
    df = pd.read_csv(csv_file, sep='\t')
    
    # Extract unique person words by gender
    male_persons = sorted(df[df['Gender'] == 'male']['Person'].unique())
    female_persons = sorted(df[df['Gender'] == 'female']['Person'].unique())
    
    # Format the output
    output_text = "Male person words:\n"
    output_text += str(male_persons) + "\n\n"
    output_text += "Female person words:\n"
    output_text += str(female_persons)
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(output_text)
    
    print(f"Extracted {len(male_persons)} male and {len(female_persons)} female person words.")
    print(f"Results written to {output_file}")
    
    # Also return the lists for further use if needed
    return male_persons, female_persons

# # Example usage
# if __name__ == "__main__":
#     # Replace these with your actual file paths
#     input_csv = "../BEC-Pro/BEC-Pro_DE.tsv"
#     output_txt = "person_words_german.txt"
    
#     male_list, female_list = extract_person_words_by_gender(input_csv, output_txt)

import pandas as pd
import re
import ast

import pandas as pd
import re
import ast

def count_person_word_occurrences(tsv_file, person_words_file, output_file):
    """
    Count occurrences of person words in the 'Text' column of a TSV file
    and update the person words file with counts.
    
    Args:
        tsv_file (str): Path to the TSV file
        person_words_file (str): Path to the file containing lists of person words
        output_file (str): Path to save the updated person words with counts
    """
    # Read the TSV file
    df = pd.read_csv(tsv_file, sep='\t')
    
    # Read the person words file
    with open(person_words_file, 'r') as f:
        content = f.read()
    
    # Extract male and female person word lists from the file
    male_list_str = re.search(r'Male person words:\n(.*?)\n\n', content, re.DOTALL).group(1)
    female_list_str = re.search(r'Female person words:\n(.*)', content, re.DOTALL).group(1)
    
    # Convert string representations to actual lists
    male_persons = ast.literal_eval(male_list_str)
    female_persons = ast.literal_eval(female_list_str)
    
    # Count occurrences of each person word in the 'Text' column
    male_counts = {person: 0 for person in male_persons}
    female_counts = {person: 0 for person in female_persons}
    
    # Combine all text from the 'Text' column
    all_text = ' '.join(df['Text'].astype(str).tolist())
    
    # Count occurrences for male person words
    for person in male_persons:
        # Use word boundary to match whole words only
        # Case insensitive search
        pattern = r'\b' + re.escape(person) + r'\b'
        matches = re.findall(pattern, all_text, re.IGNORECASE)
        male_counts[person] = len(matches)
    
    # Count occurrences for female person words
    for person in female_persons:
        pattern = r'\b' + re.escape(person) + r'\b'
        matches = re.findall(pattern, all_text, re.IGNORECASE)
        female_counts[person] = len(matches)
    
    # Calculate total counts
    total_male_occurrences = sum(male_counts.values())
    total_female_occurrences = sum(female_counts.values())
    
    # Format the output with counts
    output_text = "Male person words with occurrence counts:\n"
    male_with_counts = []
    for person in male_persons:
        male_with_counts.append(f"'{person}': {male_counts[person]}")
    output_text += "{" + ", ".join(male_with_counts) + "}\n"
    output_text += f"Total male person word occurrences: {total_male_occurrences}\n\n"
    
    output_text += "Female person words with occurrence counts:\n"
    female_with_counts = []
    for person in female_persons:
        female_with_counts.append(f"'{person}': {female_counts[person]}")
    output_text += "{" + ", ".join(female_with_counts) + "}\n"
    output_text += f"Total female person word occurrences: {total_female_occurrences}\n\n"
    
    # Add combined total
    output_text += f"Total person word occurrences (male + female): {total_male_occurrences + total_female_occurrences}"
    
    # Write to output file
    with open(output_file, 'w') as f:
        f.write(output_text)
    
    print(f"Counted occurrences of {len(male_persons)} male and {len(female_persons)} female person words.")
    print(f"Total occurrences: Male = {total_male_occurrences}, Female = {total_female_occurrences}")
    print(f"Results written to {output_file}")
    
    # Return the dictionaries and totals for further use if needed
    return {
        "male_counts": male_counts, 
        "female_counts": female_counts,
        "total_male": total_male_occurrences,
        "total_female": total_female_occurrences
    }

# Example usage
if __name__ == "__main__":
    # Replace these with your actual file paths
    input_tsv = "../Gap/gap_flipped.tsv"
    person_words_file = "../BEC-Pro/person_words_english.txt"
    output_file = "../BEC-Pro/person_words_english_counts.txt"
    
    result = count_person_word_occurrences(input_tsv, person_words_file, output_file)