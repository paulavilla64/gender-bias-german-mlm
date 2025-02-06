from transformers import AutoTokenizer, BertTokenizer

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
tokenizer_EN = BertTokenizer.from_pretrained("bert-base-uncased")


male_professions = ['Trockenbaumonteur', 'Stahlarbeiter', 'Mechaniker für mobile Geräte', 'Busmechaniker', 'Kfz-Servicetechniker', 'Heizungsmechaniker', 'Elektroinstallateur', 'Betriebsingenieur', 'Holzfäller', 'Bodenleger', 'Dachedecker', 'Bergbaumaschinentechniker', 'Elektriker', 'Kfz-Mechaniker', 'Schaffner', 'Klempner', 'Zimmermann', 'Installateur von Sicherheitssystemen', 'Maurer', 'Feuerwehrmann', 'Kindergärtner', 'Dentalhygieniker', 'Logopäde', 'Zahnarzthelfer', 'Kinderbetreuer', 'Medizintechniker', 'Sekretär', 'Arzthelfer', 'Friseur', 'Ernährungsberater',
                    'Berufskrankenpfleger', 'Betreuungslehrer', 'Rechtsanwaltsgehilfe', 'Fakturist', 'Phlebologe', 'Rezeptionist', 'Haushälter', 'staatlich geprüfter Krankenpfleger', 'Buchhalter', 'Gesundheitsberater', 'Verkäufer', 'Leiter religiöser Aktivitäten', 'Verkehrslotse', 'Fotograf', 'Bademeister', 'Herbergsverwalter', 'Heilpraktiker', 'Vertriebsmitarbeiter', 'Postbeamter', 'Elektro-Monteur', 'Versicherungskaufmann', 'Versicherungsvermittler', 'medizinischer Wissenschaftler', 'Statistiker', 'Ausbilder', 'Richter', 'Barkeeper', 'Fahrdienstleiter', 'Auftragssachbearbeiter', 'Postsortierer']

female_professions = ['Trockenbaumonteurin', 'Stahlarbeiterin', 'Mechanikerin für mobile Geräte', 'Busmechanikerin', 'Kfz-Servicetechnikerin', 'Heizungsmechanikerin', 'Elektroinstallateurin', 'Betriebsingenieurin', 'Holzfällerin', 'Bodenlegerin', 'Dachdeckerin', 'Bergbaumaschinentechnikerin', 'Elektrikerin', 'Kfz-Mechanikerin', 'Schaffnerin', 'Klempnerin', 'Zimmerin', 'Installateurin von Sicherheitssystemen', 'Maurerin', 'Feuerwehrfrau', 'Kindergärtnerin', 'Dentalhygienikerin', 'Logopädin', 'Zahnarzthelferin', 'Kinderbetreuerin', 'Medizintechnikerin', 'Sekretärin', 'Arzthelferin', 'Friseurin', 'Ernährungsberaterin',
                      'Berufskrankenpflegerin', 'Betreuungslehrerin', 'Rechtsanwaltsgehilfin', 'Fakturistin', 'Phlebologin', 'Rezeptionistin', 'Haushälterin', 'staatlich geprüfte Krankenpflegerin', 'Buchhalterin', 'Gesundheitsberaterin', 'Verkäuferin', 'Leiterin religiöser Aktivitäten', 'Verkehrslotsin', 'Fotografin', 'Bademeisterin', 'Herbergsverwalterin', 'Heilpraktikerin', 'Vertriebsmitarbeiterin', 'Postbeamtin', 'Elektro-Monteurin', 'Versicherungskauffrau', 'Versicherungsvermittlerin', 'medizinische Wissenschaftlerin', 'Statistikerin', 'Ausbilderin', 'Richterin', 'Barkeeperin', 'Fahrdienstleiterin', 'Auftragssachbearbeiterin', 'Postsortiererin']

gender_neutral_professions = [
    'Trockenbaumontagekraft', 'Stahlarbeitskraft', 'Fachkraft für mobile Geräte', 'Mechanik Fachkraft für Busse',
    'Kfz-Servicetechnikfachkraft', 'Fachkraft für Heizungstechnik', 'Elektrofachkraft', 'Betriebsingenieur*in',
    'Fachkraft für Holz-und Bautenschutzarbeiten', 'Bodenleger*in', 'Dachdecker*in', 'Bergbaumaschinentechnikfachkraft',
    'Elektrofachkraft', 'Fachkraft für Kfz-Mechanik', 'Fachkraft in der Eisenbahn', 'Klempner*in', 'Zimmereifachkraft',
    'Installationsfachkraft für Sicherheitssysteme', 'Maurer*in', 'Einsatzkraft der Feuerwehr', 'Kita-Fachkraft',
    'Dentalhygiene Fachkraft', 'Logopäde*in', 'Zahnarzthilfskraft', 'Kinderbetreuungsperson',
    'Fachkraft für Medizintechnik', 'Sekretär*in', 'Arzthilfskraft', 'Fachkraft im Haarsalon', 'Ernährungsberatungskraft',
    'Pflegefachperson', 'Betreuungslehrkraft', 'Rechtsanwaltsfachkraft', 'Fakturist*in', 'Phlebologe*in',
    'Person am Empfang', 'Haushaltskraft', 'staatlich geprüfte Pflegefachkraft', 'Buchhaltungsfachkraft',
    'Gesundheitsberater*in', 'Verkaufskraft', 'Leitung religiöser Aktivitäten', 'Verkehrslotse*in',
    'Fotografie betreibende Person', 'Badeaufsicht', 'Herbergsverwalter*in', 'Naturheil Fachkraft',
    'Vertriebsarbeitskraft', 'Fachkraft für Kurier-, Express- und Postdienstleistungen', 'Elektromonteur*in',
    'Versicherungsfachkraft', 'Versicherungsvermittler*in', 'medizinisch forschende Person', 'Statistiker*in',
    'Ausbildungskraft', 'Recht sprechende Person', 'Servicekraft an der Bar', 'Fahrdienstleitung', 'Sachbearbeitende Person',
    'Postsortierfachkraft'
]

print("Decoded tokens:", tokenizer.convert_ids_to_tokens(11039))
print("Encoded tokens:", tokenizer.convert_tokens_to_ids("Ehemann"))

# sentence = "Trockenbaumonteurin"
# tokens = tokenizer_EN.tokenize(sentence)
# print(tokens)

sentence = "Mutter"
tokens = tokenizer.tokenize(sentence)
print(tokens)

# token_ids = tokenizer.convert_tokens_to_ids(tokens)
# print(list(zip(tokens, token_ids)))

# tokenized_professions_male = [
#     tokenizer.tokenize(prof) for prof in male_professions]

# print("Male professions:")
# for prof, tokens in zip(male_professions, tokenized_professions_male):
#     print(f"{prof}: {tokens}")

# tokenized_professions_female = [tokenizer.tokenize(prof) for prof in female_professions]

# print("Female professions:")
# for prof, tokens in zip(female_professions, tokenized_professions_female):
#     print(f"{prof}: {tokens}")

#tokenized_professions_neutral = [tokenizer.tokenize(prof) for prof in gender_neutral_professions]

print("Gender neutral professions:")
for prof in gender_neutral_professions:
    print(f"{prof}")
