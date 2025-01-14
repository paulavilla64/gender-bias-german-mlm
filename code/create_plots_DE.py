import pandas as pd
import matplotlib.pyplot as plt

# Read data
df = pd.read_csv("../data/results_DE.csv", delimiter='\t')

# Statistically Male Professions:

# Define the custom order for professions
profession_order_male = [
    'Maurer', 'Dachedecker', 'Zimmermann', 'Bodenleger', 'Elektroinstallateur', 'Stahlarbeiter',
    'Mechaniker für mobile Geräte', 'Kfz-Servicetechniker', 'Betriebsingenieur', 'Trockenbaumonteur', 'Bergbaumaschinentechniker',
    'Heizungsmechaniker', 'Installateur von Sicherheitssystemen', 'Klempner', 'Holzfäller', 'Elektriker',
    'Busmechaniker', 'Schaffner', 'Feuerwehrmann', 'Kfz-Mechaniker'
]

profession_order_male_female_version = [
    'Maurerin', 'Dachdeckerin', 'Zimmerin', 'Bodenlegerin', 'Elektroinstallateurin', 'Stahlarbeiterin',
    'Mechanikerin für mobile Geräte', 'Kfz-Servicetechnikerin', 'Betriebsingenieurin', 'Trockenbaumonteurin', 'Bergbaumaschinentechnikerin',
    'Heizungsmechanikerin', 'Installateurin von Sicherheitssystemen', 'Klempnerin', 'Holzfällerin', 'Elektrikerin',
    'Busmechanikerin', 'Schaffnerin', 'Feuerwehrfrau', 'Kfz-Mechanikerin'
]

# Define the English labels for x-axis
labels = [
    'mason', 'roofer', 'carpenter', 'floor installer', 'electrical installer', 'steel worker',
    'mobile equipment mechanic', 'service technician', 'operating engineer', 'taper', 'mining machine operator',
    'heating mechanic', 'security system installer', 'plumber', 'logging worker', 'electrician',
    'bus mechanic', 'conductor', 'firefighter', 'repairer'
]

# Combine male and female versions into one order
profession_order_combined = list(
    sum(zip(profession_order_male, profession_order_male_female_version), ())
)

# Filter rows for statistically female professions and their male counterparts
female_professions_1 = df[df['Profession'].isin(
    profession_order_male_female_version)]
male_professions_1 = df[df['Profession'].isin(
    profession_order_male)]

# Add 'Form' column to differentiate male and female
male_professions_1['Form'] = 'Male'
female_professions_1['Form'] = 'Female'

# Combine male and female datasets
combined_df = pd.concat([female_professions_1, male_professions_1])

# Select necessary columns
columns_needed = ['Profession', 'Form', 'Pre_Assoc']
combined_df = combined_df[columns_needed]

# Group and calculate mean
grouped = combined_df.groupby(['Profession', 'Form']).mean().reset_index()

# Pivot for plotting
pivot_data = grouped.pivot(
    index='Profession', columns='Form', values='Pre_Assoc')

# Reorder the DataFrame according to the combined profession order
pivot_data = pivot_data.reindex(profession_order_combined)

# Plot the data
fig, ax = plt.subplots(figsize=(15, 7))

# Set title for the plot
ax.set_title("Statistically Male Professions German", fontsize=16)

# Define bar positions and width
bar_width = 0.4

positions = range(len(labels))

# Plot bars for male and female
ax.bar([p - bar_width / 2 for p in positions], pivot_data.loc[profession_order_male_female_version, 'Female'],
       width=bar_width, color='orange', label='Female', zorder=3)
ax.bar([p + bar_width / 2 for p in positions], pivot_data.loc[profession_order_male, 'Male'],
       width=bar_width, color='blue', label='Male', zorder=3)
ax.set_facecolor('#f0f0f0')  # Grey-ish background for the plot

# Set x-axis ticks and labels
ax.set_xticks(positions)
ax.set_xticklabels(labels, rotation=90, ha='center')

# Set y-axis label
ax.set_ylabel('Before Fine-tuning', fontsize=12, labelpad=15)

# Set y-axis limits
ax.set_ylim(-0.25, 3)

# Add grid and adjust plot background
ax.set_axisbelow(True)

ax.grid(axis='y', color='white', linewidth=1.0, zorder=1)
ax.grid(axis='x', color='white', linewidth=1.0, zorder=1)

# Add the legend
ax.legend(loc='upper left')

# Adjust layout
plt.tight_layout()

# Save plot
plt.savefig('../data/plots/statistically_male_professions_DE.png',
            bbox_inches='tight')

# Show the plot
plt.show()


# Statistically Female Professions:

# Define the custom order for professions
profession_order_female = [
    'Ernährungsberaterin', 'staatlich geprüfte Krankenpflegerin', 'Berufskrankenpflegerin', 'Gesundheitsberaterin', 'Arzthelferin', 'Zahnarzthelferin',
    'Sekretärin', 'Kinderbetreuerin', 'Fakturistin', 'Friseurin', 'Logopädin',
    'Kindergärtnerin', 'Betreuungslehrerin', 'Rechtsanwaltsgehilfin', 'Buchhalterin', 'Dentalhygienikerin',
    'Phlebologin', 'Medizintechnikerin', 'Haushälterin', 'Rezeptionistin'
]

profession_order_female_male_version = [
    'Ernährungsberater', 'staatlich geprüfter Krankenpfleger', 'Berufskrankenpfleger', 'Gesundheitsberater', 'Arzthelfer', 'Zahnarzthelfer',
    'Sekretär', 'Kinderbetreuer', 'Fakturist', 'Friseur', 'Logopäde',
    'Kindergärtner', 'Betreuungslehrer', 'Rechtsanwaltsgehilfe', 'Buchhalter', 'Dentalhygieniker',
    'Phlebologe', 'Medizintechniker', 'Haushälter', 'Rezeptionist'
]

# Define the English labels for x-axis
labels_female = [
    'dietitian', 'registered nurse', 'vocational nurse', 'health aide', 'medical assistant', 'dental assistant',
    'secretary', 'childcare worker', 'billing clerk', 'hairdresser', 'speech-language pathologist',
    'kindergarten teacher', 'teacher assistant', 'paralegal', 'bookkeeper', 'dental hygienist',
    'phlebotomist', 'medial records technician', 'housekeeper', 'receptionist'
]

# Combine male and female versions into one order
profession_order_combined = list(
    sum(zip(profession_order_female, profession_order_female_male_version), ())
)

# Filter rows for statistically female professions and their male counterparts
female_professions = df[df['Profession'].isin(profession_order_female)]
male_professions = df[df['Profession'].isin(
    profession_order_female_male_version)]

# Add 'Form' column to differentiate male and female
male_professions['Form'] = 'Male'
female_professions['Form'] = 'Female'

# Combine male and female datasets
combined_df = pd.concat([female_professions, male_professions])

# Select necessary columns
columns_needed = ['Profession', 'Form', 'Pre_Assoc']
combined_df = combined_df[columns_needed]

# Group and calculate mean
grouped = combined_df.groupby(['Profession', 'Form']).mean().reset_index()

# Pivot for plotting
pivot_data = grouped.pivot(
    index='Profession', columns='Form', values='Pre_Assoc')

# Reorder the DataFrame according to the combined profession order
pivot_data = pivot_data.reindex(profession_order_combined)

# Plot the data
fig, ax = plt.subplots(figsize=(15, 7))

# Set title for the plot
ax.set_title("Statistically Female Professions German", fontsize=16)

# Define bar positions and width
bar_width = 0.4

positions = range(len(labels_female))

# Plot bars for male and female
ax.bar([p - bar_width / 2 for p in positions], pivot_data.loc[profession_order_female, 'Female'],
       width=bar_width, color='orange', label='Female', zorder=3)
ax.bar([p + bar_width / 2 for p in positions], pivot_data.loc[profession_order_female_male_version, 'Male'],
       width=bar_width, color='blue', label='Male', zorder=3)
ax.set_facecolor('#f0f0f0')  # Grey-ish background for the plot

# Set x-axis ticks and labels
ax.set_xticks(positions)
ax.set_xticklabels(labels_female, rotation=90, ha='center')

# Set y-axis label
ax.set_ylabel('Before Fine-tuning', fontsize=12, labelpad=15)

# Set y-axis limits
ax.set_ylim(-0.25, 3)

# Add grid and adjust plot background
ax.set_axisbelow(True)

ax.grid(axis='y', color='white', linewidth=1.0, zorder=1)
ax.grid(axis='x', color='white', linewidth=1.0, zorder=1)

# Add the legend
ax.legend(loc='upper left')

# Adjust layout
plt.tight_layout()

# Save plot
plt.savefig('../data/plots/statistically_female_professions_DE.png',
            bbox_inches='tight')

# Show the plot
plt.show()


# Statistically Balanced Professions:

# Define the custom order for professions
profession_order_balanced = [
    'Ausbilderin', 'Vertriebsmitarbeiterin', 'Verkäuferin', 'Fotografin', 'Richterin', 'Heilpraktikerin',
    'medizinische Wissenschaftlerin', 'Barkeeperin', 'Postbeamtin', 'Verkehrslotsin', 'Herbergsverwalterin',
    'Statistikerin', 'Versicherungsvermittlerin', 'Elektro-Monteurin', 'Postsortiererin', 'Leiterin religiöser Aktivitäten',
    'Auftragssachbearbeiterin', 'Fahrdienstleiterin', 'Versicherungskauffrau', 'Bademeisterin'
]

profession_order_balanced_male_version = [
    'Ausbilder', 'Vertriebsmitarbeiter', 'Verkäufer', 'Fotograf', 'Richter', 'Heilpraktiker',
    'medizinischer Wissenschaftler', 'Barkeeper', 'Postbeamter', 'Verkehrslotse', 'Herbergsverwalter',
    'Statistiker', 'Versicherungsvermittler', 'Elektro-Monteur', 'Postsortierer', 'Leiter religiöser Aktivitäten',
    'Auftragssachbearbeiter', 'Fahrdienstleiter', 'Versicherungskaufmann', 'Bademeister'
]

# Define the English labels for x-axis
labels_balanced = [
    'training specialist', 'sales agent', 'salesperson', 'photographer', 'judge', 'healthcare practitioner',
    'medical scientist', 'bartender', 'mail clerk', 'crossing guard', 'lodging manager',
    'statistician', 'insurance underwriter', 'electrical assembler', 'mail sorter', 'director of religious activities',
    'order clerk', 'dispatcher', 'insurance sales agent', 'lifeguard'
]

# Combine male and female versions into one order
profession_order_combined = list(
    sum(zip(profession_order_balanced, profession_order_balanced_male_version), ())
)

# Filter rows for statistically female professions and their male counterparts
female_professions_2 = df[df['Profession'].isin(profession_order_balanced)]
male_professions_2 = df[df['Profession'].isin(
    profession_order_balanced_male_version)]

# Add 'Form' column to differentiate male and female
male_professions_2['Form'] = 'Male'
female_professions_2['Form'] = 'Female'

# Combine male and female datasets
combined_df = pd.concat([female_professions_2, male_professions_2])

# Select necessary columns
columns_needed = ['Profession', 'Form', 'Pre_Assoc']
combined_df = combined_df[columns_needed]

# Group and calculate mean
grouped = combined_df.groupby(['Profession', 'Form']).mean().reset_index()

# Pivot for plotting
pivot_data = grouped.pivot(
    index='Profession', columns='Form', values='Pre_Assoc')

# Reorder the DataFrame according to the combined profession order
pivot_data = pivot_data.reindex(profession_order_combined)

# Plot the data
fig, ax = plt.subplots(figsize=(15, 7))

# Set title for the plot
ax.set_title("Statistically Balanced Professions German", fontsize=16)

# Define bar positions and width
bar_width = 0.4

positions = range(len(labels_balanced))

# Plot bars for male and female
ax.bar([p - bar_width / 2 for p in positions], pivot_data.loc[profession_order_balanced, 'Female'],
       width=bar_width, color='orange', label='Female', zorder=3)
ax.bar([p + bar_width / 2 for p in positions], pivot_data.loc[profession_order_balanced_male_version, 'Male'],
       width=bar_width, color='blue', label='Male', zorder=3)
ax.set_facecolor('#f0f0f0')  # Grey-ish background for the plot

# Set x-axis ticks and labels
ax.set_xticks(positions)
ax.set_xticklabels(labels_balanced, rotation=90, ha='center')

# Set y-axis label
ax.set_ylabel('Before Fine-tuning', fontsize=12, labelpad=15)

# Set y-axis limits
ax.set_ylim(-0.25, 3)

# Add grid and adjust plot background
ax.set_axisbelow(True)

ax.grid(axis='y', color='white', linewidth=1.0, zorder=1)
ax.grid(axis='x', color='white', linewidth=1.0, zorder=1)

# Add the legend
ax.legend(loc='upper left')

# Adjust layout
plt.tight_layout()

# Save plot
plt.savefig('../data/plots/statistically_balanced_professions_DE.png',
            bbox_inches='tight')

# Show the plot
plt.show()
