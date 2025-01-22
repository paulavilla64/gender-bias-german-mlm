# check gender marking…for all professions in the corpus, check if the corresponding person word is more likely
# (e.g., “MASK ist Lehrer” → we are expecting male person words)
# → it works…why then are the female associations higher?
# → it does not work…check across models (for future)
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import torch.nn.functional as F

device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
model = AutoModelForMaskedLM.from_pretrained("dbmdz/bert-base-german-cased")
print("loading german bert")

male_professions = ['Kindergärtner', 'Dentalhygieniker', 'Logopäde', 'Zahnarzthelfer', 'Kinderbetreuer',  'Medizintechniker', 'Sekretär', 'Arzthelfer', 'Friseur', 'Ernährungsberater',  'Berufskrankenpfleger', 'Betreuungslehrer', 'Rechtsanwaltsgehilfe', 'Fakturist',  'Phlebologe', 'Rezeptionist', 'Haushälter', 'staatlich geprüfter Krankenpfleger',  'Buchhalter', 'Gesundheitsberater', 'Trockenbaumonteur', 'Stahlarbeiter',  'Mechaniker für mobile Geräte', 'Busmechaniker', 'Kfz-Servicetechniker',  'Heizungsmechaniker', 'Elektroinstallateur', 'Betriebsingenieur', 'Holzfäller',
                    'Bodenleger', 'Dachedecker', 'Bergbaumaschinentechniker', 'Elektriker',  'Kfz-Mechaniker', 'Schaffner', 'Klempner', 'Zimmermann', 'Installateur von Sicherheitssystemen',  'Maurer', 'Feuerwehrmann', 'Verkäufer', 'Leiter religiöser Aktivitäten', 'Verkehrslotse',  'Fotograf', 'Bademeister', 'Herbergsverwalter', 'Heilpraktiker', 'Vertriebsmitarbeiter',  'Postbeamter', 'Elektro-Monteur', 'Versicherungskaufmann', 'Versicherungsvermittler',  'medizinischer Wissenschaftler', 'Statistiker', 'Ausbilder', 'Richter', 'Barkeeper',  'Fahrdienstleiter', 'Auftragssachbearbeiter', 'Postsortierer']
female_professions = ['Kindergärtnerin', 'Dentalhygienikerin', 'Logopädin', 'Zahnarzthelferin', 'Kinderbetreuerin',  'Medizintechnikerin', 'Sekretärin', 'Arzthelferin', 'Friseurin', 'Ernährungsberaterin',  'Berufskrankenpflegerin', 'Betreuungslehrerin', 'Rechtsanwaltsgehilfin', 'Fakturistin',  'Phlebologin', 'Rezeptionist', 'Haushälterin', 'staatlich geprüfte Krankenpflegerin',  'Buchhalterin', 'Gesundheitsberaterin', 'Trockenbaumonteurin', 'Stahlarbeiterin',  'Mechanikerin für mobile Geräte', 'Busmechanikerin', 'Kfz-Servicetechnikerin',  'Heizungsmechanikerin', 'Elektroinstallateurin', 'Betriebsingenieurin', 'Holzfällerin',
                      'Bodenlegerin', 'Dachdeckerin', 'Bergbaumaschinentechnikerin', 'Elektrikerin',  'Kfz-Mechaniker', 'Schaffnerin', 'Klempnerin', 'Zimmerin', 'Installateurin von Sicherheitssystemen',  'Maurerin', 'Feuerwehrfrau', 'Verkäuferin', 'Leiterin religiöser Aktivitäten', 'Verkehrslotsin',  'Fotografin', 'Bademeisterin', 'Herbergsverwalterin', 'Heilpraktikerin', 'Vertriebsmitarbeiterin',  'Postbeamtin', 'Elektro-Monteurin', 'Versicherungskauffrau', 'Versicherungsvermittlerin',  'medizinische Wissenschaftlerin', 'Statistikerin', 'Ausbilderin', 'Richterin', 'Barkeeperin',  'Fahrdienstleiterin', 'Auftragssachbearbeiterin', 'Postsortiererin']

text_templates = [
    "{} ist {}.",
    "{} arbeitet als {}.",
    "{} hat sich auf die Stelle als {} beworben.",
    "{}, ART {}, hatte einen guten Arbeitstag.",
    "{} will {} werden."
]

text_templates_mask = [
    "[MASK] ist {}.",
    "[MASK] arbeitet als {}.",
    "[MASK] hat sich auf die Stelle als {} beworben.",
    "[MASK], ART {}, hatte einen guten Arbeitstag.",
    "[MASK] will {} werden."
]

# Get token IDs for "Er" and "er"
token_ids_er = [
    tokenizer.convert_tokens_to_ids("Er"),
    tokenizer.convert_tokens_to_ids("er")
]

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

print("Probabilites for male form:")

# Iterate over all professions and templates
for profession in male_professions:
    for template in text_templates_mask:
        # Fill the template with the current profession
        # Substitute placeholders in the template
        text_mask = substitute_template(template, profession, gender="male")

        # Tokenize the text
        inputs = tokenizer(text_mask, return_tensors="pt")

        # Get logits for the [MASK] token
        outputs = model(**inputs)
        logits = outputs.logits

        # Find the index of the [MASK] token
        mask_token_indices = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

        for idx in mask_token_indices[:1]:
            # Extract logits for the current [MASK] token
            mask_token_logits = logits[0, idx, :]

            # Convert logits to probabilities
            probs = F.softmax(mask_token_logits, dim=-1)

            # Sum probabilities for "Er"/"er" and "Sie"/"sie"
            original_token_prob_er = sum(probs[token_id].item() for token_id in token_ids_er)
            original_token_prob_sie = sum(probs[token_id].item() for token_id in token_ids_sie)

        # Print results for the current profession and sentence
        print(f"Sentence: {text_mask}")
        print(f"Probability of 'Er/er': {original_token_prob_er:.4f}")
        print(f"Probability of 'Sie/sie': {original_token_prob_sie:.4f}")
        print("-" * 40)

print("Probabilites for female form:")

# Iterate over all professions and templates
for profession in female_professions:
    for template in text_templates_mask:
        # Fill the template with the current profession
        # Substitute placeholders in the template
        text_mask = substitute_template(template, profession, gender="female")

        # Tokenize the text
        inputs = tokenizer(text_mask, return_tensors="pt")

        # Get logits for the [MASK] token
        outputs = model(**inputs)
        logits = outputs.logits

        # Find the index of the [MASK] token
        mask_token_indices = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

        for idx in mask_token_indices[:1]:
            # Extract logits for the current [MASK] token
            mask_token_logits = logits[0, idx, :]

            # Convert logits to probabilities
            probs = F.softmax(mask_token_logits, dim=-1)

            # Sum probabilities for "Er"/"er" and "Sie"/"sie"
            original_token_prob_er = sum(probs[token_id].item() for token_id in token_ids_er)
            original_token_prob_sie = sum(probs[token_id].item() for token_id in token_ids_sie)

        # Print results for the current profession and sentence
        print(f"Sentence: {text_mask}")
        print(f"Probability of 'Er/er': {original_token_prob_er:.4f}")
        print(f"Probability of 'Sie/sie': {original_token_prob_sie:.4f}")
        print("-" * 40)






