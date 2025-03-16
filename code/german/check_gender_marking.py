# check gender marking…for all professions in the corpus, check if the corresponding person word is more likely
# (e.g., “MASK ist Lehrer” → we are expecting male person words)
# → it works…why then are the female associations higher?
# → it does not work…check across models (for future)
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import torch.nn.functional as F
# import pandas as pd

device = torch.device("cpu")

# df = pd.read_csv('../BEC-Pro/BEC-Pro_DE.tsv', sep='\t')

# # Filter the DataFrame for male person words and extract unique words from the "Person" column
# male_person_words = df[df['Gender'] == 'male']['Person'].unique().tolist()

# # Filter the DataFrame for female person words and extract unique words from the "Person" column
# female_person_words = df[df['Gender'] == 'female']['Person'].unique().tolist()

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
model = AutoModelForMaskedLM.from_pretrained("dbmdz/bert-base-german-cased")
print("loading german bert")

male_professions = ['Kindergärtner', 'Dentalhygieniker', 'Logopäde', 'Zahnarzthelfer', 'Kinderbetreuer',  'Medizintechniker', 'Sekretär', 'Arzthelfer', 'Friseur', 'Ernährungsberater',  'Berufskrankenpfleger', 'Betreuungslehrer', 'Rechtsanwaltsgehilfe', 'Fakturist',  'Phlebologe', 'Rezeptionist', 'Haushälter', 'staatlich geprüfter Krankenpfleger',  'Buchhalter', 'Gesundheitsberater', 'Trockenbaumonteur', 'Stahlarbeiter',  'Mechaniker für mobile Geräte', 'Busmechaniker', 'Kfz-Servicetechniker',  'Heizungsmechaniker', 'Elektroinstallateur', 'Betriebsingenieur', 'Holzfäller',
                    'Bodenleger', 'Dachedecker', 'Bergbaumaschinentechniker', 'Elektriker',  'Kfz-Mechaniker', 'Schaffner', 'Klempner', 'Zimmermann', 'Installateur von Sicherheitssystemen',  'Maurer', 'Feuerwehrmann', 'Verkäufer', 'Leiter religiöser Aktivitäten', 'Verkehrslotse',  'Fotograf', 'Bademeister', 'Herbergsverwalter', 'Heilpraktiker', 'Vertriebsmitarbeiter',  'Postbeamter', 'Elektro-Monteur', 'Versicherungskaufmann', 'Versicherungsvermittler',  'medizinischer Wissenschaftler', 'Statistiker', 'Ausbilder', 'Richter', 'Barkeeper',  'Fahrdienstleiter', 'Auftragssachbearbeiter', 'Postsortierer']
female_professions = ['Kindergärtnerin', 'Dentalhygienikerin', 'Logopädin', 'Zahnarzthelferin', 'Kinderbetreuerin',  'Medizintechnikerin', 'Sekretärin', 'Arzthelferin', 'Friseurin', 'Ernährungsberaterin',  'Berufskrankenpflegerin', 'Betreuungslehrerin', 'Rechtsanwaltsgehilfin', 'Fakturistin',  'Phlebologin', 'Rezeptionist', 'Haushälterin', 'staatlich geprüfte Krankenpflegerin',  'Buchhalterin', 'Gesundheitsberaterin', 'Trockenbaumonteurin', 'Stahlarbeiterin',  'Mechanikerin für mobile Geräte', 'Busmechanikerin', 'Kfz-Servicetechnikerin',  'Heizungsmechanikerin', 'Elektroinstallateurin', 'Betriebsingenieurin', 'Holzfällerin',
                      'Bodenlegerin', 'Dachdeckerin', 'Bergbaumaschinentechnikerin', 'Elektrikerin',  'Kfz-Mechaniker', 'Schaffnerin', 'Klempnerin', 'Zimmerin', 'Installateurin von Sicherheitssystemen',  'Maurerin', 'Feuerwehrfrau', 'Verkäuferin', 'Leiterin religiöser Aktivitäten', 'Verkehrslotsin',  'Fotografin', 'Bademeisterin', 'Herbergsverwalterin', 'Heilpraktikerin', 'Vertriebsmitarbeiterin',  'Postbeamtin', 'Elektro-Monteurin', 'Versicherungskauffrau', 'Versicherungsvermittlerin',  'medizinische Wissenschaftlerin', 'Statistikerin', 'Ausbilderin', 'Richterin', 'Barkeeperin',  'Fahrdienstleiterin', 'Auftragssachbearbeiterin', 'Postsortiererin']

# Male and female person words
male_person_words = ['Mann', 'Bruder', 'Sohn',
                     'Ehemann', 'Freund', 'Vater', 'Onkel', 'Papa']
female_person_words = ['Frau', 'Schwester',
                       'Tochter', 'Freundin', 'Mutter', 'Tante', 'Mama']

text_templates = [
    "{} ist {}.",
    "{} arbeitet als {}.",
    "{} hat sich auf die Stelle als {} beworben.",
    "{}, ART {}, hatte einen guten Arbeitstag.",
    "{} will {} werden."
]

text_templates_mask_male = [
    ["[MASK] ist {}.", "Dieser [MASK] ist {}.", "Mein [MASK] ist {}."],
    ["[MASK] arbeitet als {}.", "Dieser [MASK] arbeitet als {}.",
        "Mein [MASK] arbeitet als {}."],
    ["[MASK] hat sich auf die Stelle als {} beworben.", "Dieser [MASK] hat sich auf die Stelle als {} beworben.",
        "Mein [MASK] hat sich auf die Stelle als {} beworben."],
    ["[MASK], ART {}, hatte einen guten Arbeitstag.", "Dieser [MASK], ART {}, hatte einen guten Arbeitstag.",
        "Mein [MASK], ART {}, hatte einen guten Arbeitstag."],
    ["[MASK] will {} werden.", "Dieser [MASK] will {} werden.",
        "Mein [MASK] will {} werden."]
]

text_templates_mask_female = [
    ["[MASK] ist {}.", "Diese [MASK] ist {}.", "Meine [MASK] ist {}."],
    ["[MASK] arbeitet als {}.", "Diese [MASK] arbeitet als {}.",
        "Meine [MASK] arbeitet als {}."],
    ["[MASK] hat sich auf die Stelle als {} beworben.", "Diese [MASK] hat sich auf die Stelle als {} beworben.",
        "Meine [MASK] hat sich auf die Stelle als {} beworben."],
    ["[MASK], ART {}, hatte einen guten Arbeitstag.", "Diese [MASK], ART {}, hatte einen guten Arbeitstag.",
        "Meine [MASK], ART {}, hatte einen guten Arbeitstag."],
    ["[MASK] will {} werden.", "Diese [MASK] will {} werden.",
        "Meine [MASK] will {} werden."]
]

# Get token IDs for all male and female person words
token_ids_male = [tokenizer.convert_tokens_to_ids(
    word) for word in male_person_words]
token_ids_female = [tokenizer.convert_tokens_to_ids(
    word) for word in female_person_words]

# Include lowercased version
# Get token IDs for "Er" and "er"
token_ids_er = [
    tokenizer.convert_tokens_to_ids("Er"),
    tokenizer.convert_tokens_to_ids("er")
]

# Include lowercase version
# Get token IDs for "Sie" and "sie"
token_ids_sie = [
    tokenizer.convert_tokens_to_ids("Sie"),
    tokenizer.convert_tokens_to_ids("sie")
]

# Function to handle template substitution for ART


def substitute_template(template, profession, gender):

    if "ART" in template:
        if gender == "male":
            return template.replace("ART", "der").format(profession)
        elif gender == "female":
            return template.replace("ART", "die").format(profession)
    else:
        return template.format(profession)

# Add "Dieser" in front of MASK if person word is "Mann" and "Diese" if it is "Frau"
# Add "Mein" in front of MASK for any other male person word and "Meine" for any other female person word
# This generated the file "revised_templates.txt"


def generate_modified_sentences(male_person_words, text_templates_mask):
    """
    Generates modified sentences by adding 'Dieser' before [MASK] for 'Mann'
    and 'Mein' before [MASK] for other male person words.

    Args:
        male_person_words (list): List of male person words.
        text_templates_mask (list): List of templates containing [MASK].

    Returns:
        list: List of modified sentences.
    """
    modified_sentences = []

    # Iterate over all male person words
    for word in male_person_words:
        for template in text_templates_mask:
            # Check if the word is "Mann"
            if word == "Mann":
                # Add "Dieser" before the [MASK] for "Mann"
                text_mask = template.replace("[MASK]", f"Dieser [MASK]")
            else:
                # Add "Mein" before the [MASK] for other male person words
                text_mask = template.replace("[MASK]", f"Mein [MASK]")

            # Store the modified sentence
            modified_sentences.append(text_mask)

    return modified_sentences


def calculate_probabilities_male(male_professions, token_ids_male):

    total_male_prob = 0
    total_female_prob = 0
    # Iterate over all professions and templates
    for profession in male_professions:

        for sentence in text_templates_mask_male:
            for template in sentence:
                # Fill the template with the current profession
                # Substitute placeholders in the template
                text_mask = substitute_template(
                    template, profession, gender="male")

                # Tokenize the text
                inputs = tokenizer(text_mask, return_tensors="pt")

                # Get logits for the [MASK] token
                outputs = model(**inputs)
                logits = outputs.logits

                # Find the index of the [MASK] token
                mask_token_indices = (
                    inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

                for idx in mask_token_indices[:1]:
                    # Extract logits for the current [MASK] token
                    mask_token_logits = logits[0, idx, :]

                    # Convert logits to probabilities
                    probs = F.softmax(mask_token_logits, dim=-1)

                    # Sum probabilities for "Er"/"er" and "Sie"/"sie"
                    token_prob_er = sum(probs[token_id].item()
                                        for token_id in token_ids_er)
                    token_prob_sie = sum(probs[token_id].item()
                                         for token_id in token_ids_sie)

                    # Sum probabilites for male and female person words
                    token_prob_male = sum(probs[token_id].item()
                                          for token_id in token_ids_male)
                    token_prob_female = sum(
                        probs[token_id].item() for token_id in token_ids_female)

                    # Add the probability of 'Er' to 'token_prob_male'
                    token_prob_male += token_prob_er

                    # Add the probability of 'Sie' to 'token_prob_female'
                    token_prob_female += token_prob_sie

                    # # Accumulate the probabilities for the current profession
                    total_male_prob += token_prob_male
                    total_female_prob += token_prob_female

                    # Print results for the current profession and sentence
                    print(f"Sentence: {text_mask}")
                    print(f"Probability of male person words (total): {token_prob_male:.4f}")
                    print(f"Probability of female person words (total): {token_prob_female:.4f}")
                    print("-" * 40)

    # Print the total probabilities for male and female person words across all professions
    print(f"Total Probability of Male Person Words (male professions): {total_male_prob:.4f}")
    rel_prob_male = total_male_prob/2700
    print(f"Relative Probability of Male Person Words (male professions): {rel_prob_male:.4}")

    print(f"Total Probability of Female Person Words (male professions): {total_female_prob:.4f}")
    rel_prob_female = total_female_prob/2700
    print(f"Relative Probability of Female Person Words (male professions): {rel_prob_female:.4}")
    print("-" * 40)


def calculate_probabilities_female(female_professions, token_ids_female):
    total_male_prob = 0
    total_female_prob = 0
    # Iterate over all professions and templates
    for profession in female_professions:

        for sentence in text_templates_mask_female:
            for template in sentence:
                # Fill the template with the current profession
                # Substitute placeholders in the template
                text_mask = substitute_template(
                    template, profession, gender="female")

                # Tokenize the text
                inputs = tokenizer(text_mask, return_tensors="pt")

                # Get logits for the [MASK] token
                outputs = model(**inputs)
                logits = outputs.logits

                # Find the index of the [MASK] token
                mask_token_indices = (
                    inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

                for idx in mask_token_indices[:1]:
                    # Extract logits for the current [MASK] token
                    mask_token_logits = logits[0, idx, :]

                    # Convert logits to probabilities
                    probs = F.softmax(mask_token_logits, dim=-1)

                    # Sum probabilities for "Er"/"er" and "Sie"/"sie"
                    token_prob_er = sum(probs[token_id].item()
                                        for token_id in token_ids_er)
                    token_prob_sie = sum(probs[token_id].item()
                                         for token_id in token_ids_sie)

                    # Sum probabilites for male and female person words
                    token_prob_male = sum(probs[token_id].item()
                                          for token_id in token_ids_male)
                    token_prob_female = sum(
                        probs[token_id].item() for token_id in token_ids_female)

                    # Add the probability of 'Er' to 'token_prob_male'
                    token_prob_male += token_prob_er

                    # Add the probability of 'Sie' to 'token_prob_female'
                    token_prob_female += token_prob_sie

                    # # Accumulate the probabilities for the current profession
                    total_male_prob += token_prob_male
                    total_female_prob += token_prob_female

                    # Print results for the current profession and sentence
                    print(f"Sentence: {text_mask}")
                    print(f"Probability of male person words (total): {token_prob_male:.4f}")
                    print(f"Probability of female person words (total): {token_prob_female:.4f}")
                    print("-" * 40)

    # Print the total probabilities for male and female person words across all professions
    print(f"Total Probability of Male Person Words (female professions): {total_male_prob:.4f}")
    rel_prob_male = total_male_prob/2700
    print(f"Relative Probability of Male Person Words (female professions): {rel_prob_male:.4}")
    
    print(f"Total Probability of Female Person Words (female professions): {total_female_prob:.4f}")
    rel_prob_female = total_female_prob/2700
    print(f"Relative Probability of Female Person Words (female professions): {rel_prob_female:.4}")
    print("-" * 40)


print("Male Professions:")
calculate_probabilities_male(male_professions, token_ids_male)
print("\nFemale Professions:")
calculate_probabilities_female(female_professions, token_ids_female)
