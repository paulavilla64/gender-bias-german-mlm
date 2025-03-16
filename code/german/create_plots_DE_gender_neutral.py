import pandas as pd
import matplotlib.pyplot as plt

# Read data
df = pd.read_csv("../data/google-bert/gender-neutral/google_bert_results_gender_neutral.csv", delimiter='\t')

# Inspect the data
print(df.head())
print(df.info())

profession_order_male_gender_neutral = [
    'Maurer*in', 'Dachdecker*in', 'Zimmereifachkraft', 'Bodenleger*in', 'Elektrofachkraft', 'Stahlarbeitskraft',
    'Fachkraft für mobile Geräte', 'Kfz-Servicetechnikfachkraft', 'Betriebsingenieur*in', 'Trockenbaumontagekraft', 'Bergbaumaschinentechnikfachkraft',
    'Fachkraft für Heizungstechnik', 'Installationsfachkraft für Sicherheitssysteme', 'Klempner*in', 'Fachkraft für Holz-und Bautenschutzarbeiten', 'Elektrofachkraft',
    'Mechanik Fachkraft für Busse', 'Fachkraft in der Eisenbahn', 'Einsatzkraft der Feuerwehr', 'Fachkraft für Kfz-Mechanik'
]

print("Expected Index Values:", profession_order_male_gender_neutral)


# Filter rows for statistically male professions
male_professions = df[df['Prof_Gender'] == 'male']

# Select necessary columns
columns_needed = ['Profession', 'Gender', 'Pre_Assoc']
male_professions_filtered = male_professions[columns_needed]

# Group and calculate mean
grouped = male_professions_filtered.groupby(
    ['Profession', 'Gender']).mean().reset_index()

# Pivot for plotting
pre_data_male = grouped.pivot(
    index='Profession', columns='Gender', values='Pre_Assoc')

# Reorder the DataFrame according to the custom profession order
pre_data_male = pre_data_male.loc[profession_order_male_gender_neutral]

# Define fixed y-limits
y_min_male, y_max_male = -1.25, 1.5

# Plot the data
fig, ax = plt.subplots(figsize=(12, 8))

# Set title for the first plot
ax.set_title("Google Bert - Statistically Male Professions Gender Neutral", fontsize=16)

# Pre-association plot
pre_data_male.plot(kind='bar', color=['orange', 'blue'], legend=False, zorder=3, ax=ax)

ax.set_ylabel('Before fine-tuning', fontsize=12, labelpad=15)
ax.set_facecolor('#f0f0f0')  # Grey-ish background for the plot
ax.set_xticklabels(pre_data_male.index, rotation=45, ha='right')
# Set the same y-limits for both plots
ax.set_ylim([y_min_male, y_max_male])

# Add white grid lines (y and x axes)
ax.grid(axis='y', color='white', linewidth=1.0, zorder=1)
ax.grid(axis='x', color='white', linewidth=1.0, zorder=1)

# Add the legend below the plots
fig.legend(['Female', 'Male'], loc='lower left',
           bbox_to_anchor=(0.5, 0.05), ncol=2)

# Adjust layout
plt.tight_layout()

# Save plot
plt.savefig("../data/plots/google_bert_statistically_male_professions_gender_neutral.png",
            bbox_inches='tight')

profession_order_female_gender_neutral = [
    'Ernährungsberatungskraft', 'staatlich geprüfte Pflegefachkraft', 'Pflegefachperson',
    'Gesundheitsberater*in', 'Arzthilfskraft', 'Zahnarzthilfskraft', 'Sekretär*in',  
    'Kinderbetreuungsperson', 'Fakturist*in', 'Fachkraft im Haarsalon', 'Logopäde*in', 
    'Kita-Fachkraft', 'Betreuungslehrkraft', 'Rechtsanwaltsfachkraft', 'Buchhaltungsfachkraft', 'Dentalhygiene Fachkraft', 
    'Phlebologe*in', 'Fachkraft für Medizintechnik','Haushaltskraft', 'Person am Empfang']

# Filter rows for statistically female professions
female_professions = df[df['Prof_Gender'] == 'female']

# Select necessary columns
columns_needed = ['Profession', 'Gender', 'Pre_Assoc']
female_professions_filtered = female_professions[columns_needed]

# Group and calculate mean
grouped = female_professions_filtered.groupby(
    ['Profession', 'Gender']).mean().reset_index()

# Pivot for plotting
pre_data_female = grouped.pivot(
    index='Profession', columns='Gender', values='Pre_Assoc')

# Reorder the DataFrame according to the custom profession order
pre_data_female = pre_data_female.loc[profession_order_female_gender_neutral]

# Define fixed y-limits
y_min_female, y_max_female = -1.25, 1.5

# Plot the data
fig, ax = plt.subplots(figsize=(12, 8))

# Set title for the first plot
ax.set_title("Google Bert - Statistically Female Professions Gender Neutral", fontsize=16)

# Pre-association plot
pre_data_female.plot(kind='bar', color=['orange', 'blue'], legend=False, zorder=3, ax=ax)
ax.set_ylabel('Before fine-tuning', fontsize=12, labelpad=15)
ax.set_facecolor('#f0f0f0')  # Grey-ish background for the plot
ax.set_xticklabels(pre_data_female.index, rotation=45, ha='right')
# Set the same y-limits for both plots
ax.set_ylim([y_min_female, y_max_female])

# Add white grid lines (y and x axes)
ax.grid(axis='y', color='white', linewidth=1.0, zorder=1)
ax.grid(axis='x', color='white', linewidth=1.0, zorder=1)

# Add the legend below the plots
fig.legend(['Female', 'Male'], loc='lower left',
           bbox_to_anchor=(0.5, 0.05), ncol=2)

# Adjust layout
plt.tight_layout()

# Save plot
plt.savefig("../data/plots/google_bert_statistically_female_professions_gender_neutral.png",
            bbox_inches='tight')

profession_order_balanced_gender_neutral = ['Ausbildungskraft', 'Vertriebsarbeitskraft', 'Verkaufskraft', 
    'Fotografie betreibende Person', 'Recht sprechende Person', 'Naturheil Fachkraft', 'medizinisch forschende Person',
    'Servicekraft an der Bar', 'Fachkraft für Kurier-, Express- und Postdienstleistungen', 'Verkehrslotse*in',
    'Herbergsverwalter*in', 'Statistiker*in', 'Versicherungsvermittler*in', 'Elektromonteur*in', 'Postsortierfachkraft',
    'Leitung religiöser Aktivitäten', 'Sachbearbeitende Person', 'Fahrdienstleitung', 'Versicherungsfachkraft',
    'Badeaufsicht']

# Filter rows for balanced professions
balanced_professions = df[df['Prof_Gender'] == 'balanced']

# Select necessary columns
columns_needed = ['Profession', 'Gender', 'Pre_Assoc']
balanced_professions_filtered = balanced_professions[columns_needed]

# Group and calculate mean
grouped = balanced_professions_filtered.groupby(
    ['Profession', 'Gender']).mean().reset_index()

# Pivot for plotting
pre_data_balanced = grouped.pivot(
    index='Profession', columns='Gender', values='Pre_Assoc')

print("Actual Index Values:", pre_data_balanced.index)

# Reorder the DataFrame according to the custom profession order
pre_data_balanced = pre_data_balanced.loc[profession_order_balanced_gender_neutral]

# Define fixed y-limits
y_min_balanced, y_max_balanced = -1.25, 1.5

# Plot the data
fig, ax = plt.subplots(figsize=(12, 8))

# Set title for the first plot
ax.set_title("Google Bert - Statistically Balanced Professions Gender Neutral", fontsize=16)

# Pre-association plot
pre_data_balanced.plot(kind='bar', color=['orange', 'blue'], legend=False, zorder=3, ax=ax)
ax.set_ylabel('Before fine-tuning', fontsize=12, labelpad=15)
ax.set_facecolor('#f0f0f0')  # Grey-ish background for the plot
ax.set_xticklabels(pre_data_balanced.index, rotation=45, ha='right')
# Set the same y-limits for both plots
ax.set_ylim([y_min_balanced, y_max_balanced])

# Add white grid lines (y and x axes)
ax.grid(axis='y', color='white', linewidth=1.0, zorder=1)
ax.grid(axis='x', color='white', linewidth=1.0, zorder=1)

# Add the legend below the plots
fig.legend(['Female', 'Male'], loc='lower left',
           bbox_to_anchor=(0.5, 0.05), ncol=2)

# Adjust layout
plt.tight_layout()

# Save plot
plt.savefig("../data/plots/google_bert_statistically_balanced_professions_gender_neutral.png",
            bbox_inches='tight')
plt.show()
