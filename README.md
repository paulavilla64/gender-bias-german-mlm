# Gender bias in German Masked Language Models

This repository holds the code for my master thesis entitled "Gender Bias in German Masked Language Models: Measurement, Mitigation, and the Role of Gender-Inclusive Language", written at the University of Stuttgart.

## Setup

### Environment Setup

The project uses a Conda environment for dependency management. You can activate the environment using the provided `env.yml` file:

```bash
# Create the environment from the yml file
conda env create -f env.yml

# Activate the environment
conda activate gender-bias-german-mlm

```

## Running the Code

### Perplexity-based Gender Bias Measurement

The main script for measuring gender bias using perplexity differences can be found in code/

```bash

cd code/
python measure_perplexity_for_bias_evaluation.py

```

### Configuration Options

Template Type: Change the variable `typ` to select different profession templates:

"english" - English standard profession templates
"regular" - German standard profession templates
"token_balanced" - German Token-balanced templates
"gender_neutral" - German Gender-inclusive templates

Model Selection: Modify the variable `base_model_id` to choose the BERT model:

"bert-base-uncased" - English BERT
"bert-base-german-dbmdz-cased" - DBMDZ BERT
"google-bert/bert-base-german-cased" - Google BERT
"deepset/gbert-base" - G-BERT
"distilbert/distilbert-base-german-cased" - DistilBERT


## Repository Structure

```
code/                   # Main source code
datasets/               # Dataset files and evaluation corpora
results/                # Generated results and outputs
env.yml                 # Conda environment specification
README.md               # This file
```

## Data

The repository includes datasets and evaluation corpora specifically designed for measuring and mitigating gender bias in German masked language models.

### Datasets

The `datasets/` folder contains:

```
BEC-Pro/                # Template-based corpus for measuring gender bias
Gap corpus/             # Training corpus for fine-tuning models to mitigate bias
Lou corpus/             # Additional training corpus for bias mitigation fine-tuning
ag_news/                # English validation set for English model evaluation
deu_news/               # German validation set for German model evaluation
```

## Results

Generated results, analysis outputs, and visualizations are stored in the results/ directory.