from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")


male_professions = ['Trockenbaumonteur', 'Stahlarbeiter', 'Mechaniker für mobile Geräte', 'Busmechaniker', 'Kfz-Servicetechniker', 'Heizungsmechaniker', 'Elektroinstallateur', 'Betriebsingenieur', 'Holzfäller', 'Bodenleger', 'Dachedecker', 'Bergbaumaschinentechniker', 'Elektriker', 'Kfz-Mechaniker', 'Schaffner', 'Klempner', 'Zimmermann', 'Installateur von Sicherheitssystemen', 'Maurer', 'Feuerwehrmann', 'Kindergärtner', 'Dentalhygieniker', 'Logopäde', 'Zahnarzthelfer', 'Kinderbetreuer', 'Medizintechniker', 'Sekretär', 'Arzthelfer', 'Friseur', 'Ernährungsberater', 'Berufskrankenpfleger', 'Betreuungslehrer', 'Rechtsanwaltsgehilfe', 'Fakturist', 'Phlebologe', 'Rezeptionist', 'Haushälter', 'staatlich geprüfter Krankenpfleger', 'Buchhalter', 'Gesundheitsberater', 'Verkäufer', 'Leiter religiöser Aktivitäten', 'Verkehrslotse', 'Fotograf', 'Bademeister', 'Herbergsverwalter', 'Heilpraktiker', 'Vertriebsmitarbeiter', 'Postbeamter', 'Elektro-Monteur', 'Versicherungskaufmann', 'Versicherungsvermittler', 'medizinischer Wissenschaftler', 'Statistiker', 'Ausbilder', 'Richter', 'Barkeeper', 'Fahrdienstleiter', 'Auftragssachbearbeiter', 'Postsortierer']

female_professions = ['Trockenbaumonteurin', 'Stahlarbeiterin', 'Mechanikerin für mobile Geräte', 'Busmechanikerin', 'Kfz-Servicetechnikerin', 'Heizungsmechanikerin', 'Elektroinstallateurin', 'Betriebsingenieurin', 'Holzfällerin', 'Bodenlegerin', 'Dachdeckerin', 'Bergbaumaschinentechnikerin', 'Elektrikerin', 'Kfz-Mechanikerin', 'Schaffnerin', 'Klempnerin', 'Zimmerin', 'Installateurin von Sicherheitssystemen', 'Maurerin', 'Feuerwehrfrau', 'Kindergärtnerin', 'Dentalhygienikerin', 'Logopädin', 'Zahnarzthelferin', 'Kinderbetreuerin', 'Medizintechnikerin', 'Sekretärin', 'Arzthelferin', 'Friseurin', 'Ernährungsberaterin', 'Berufskrankenpflegerin', 'Betreuungslehrerin', 'Rechtsanwaltsgehilfin', 'Fakturistin', 'Phlebologin', 'Rezeptionistin', 'Haushälterin', 'staatlich geprüfte Krankenpflegerin', 'Buchhalterin', 'Gesundheitsberaterin', 'Verkäuferin', 'Leiterin religiöser Aktivitäten', 'Verkehrslotsin', 'Fotografin', 'Bademeisterin', 'Herbergsverwalterin', 'Heilpraktikerin', 'Vertriebsmitarbeiterin', 'Postbeamtin', 'Elektro-Monteurin', 'Versicherungskauffrau', 'Versicherungsvermittlerin', 'medizinische Wissenschaftlerin', 'Statistikerin', 'Ausbilderin', 'Richterin', 'Barkeeperin', 'Fahrdienstleiterin', 'Auftragssachbearbeiterin', 'Postsortiererin']


sentence = "Ingenieur"
tokens = tokenizer.tokenize(sentence)
print(tokens)

# sentence = "Sie ist eine Schreinerin."
# tokens = tokenizer.tokenize(sentence)
# print(tokens)

# token_ids = tokenizer.convert_tokens_to_ids(tokens)
# print(list(zip(tokens, token_ids)))

tokenized_professions_male = [tokenizer.tokenize(prof) for prof in male_professions]

# print("Male professions:")
# for prof, tokens in zip(male_professions, tokenized_professions_male):
#     print(f"{prof}: {tokens}")

# tokenized_professions_female = [tokenizer.tokenize(prof) for prof in female_professions]

# print("Female professions:")
# for prof, tokens in zip(female_professions, tokenized_professions_female):
#     print(f"{prof}: {tokens}")