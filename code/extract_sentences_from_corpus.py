from datasets import load_dataset
import datasets

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
dataset = load_dataset("ag_news", split="train", download_mode="force_redownload")

# Print some example data
print(dataset)
print(dataset[0])  # Print the first record

import pandas as pd

df = pd.DataFrame(dataset)
print(df.head())

import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

# Extract sentences from the first 5 news articles
for i in range(5):
    sentences = sent_tokenize(dataset[i]["text"])
    cleaned_sentences = [s.replace("\\", "") for s in sentences]

    print(f"Article {i+1}: {cleaned_sentences}\n")

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

extract_and_save_sentences(input_file)
