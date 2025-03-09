#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
# Charger le dataset
df = pd.read_csv("/data/dataset_projet_evaluation.csv")

# Afficher les premières lignes
print(df.head())


# In[2]:


# 1: Pseudonymisation des noms
df['Nom'] = df.index.map(lambda x: f"User_{x}")

# Affichage du résultat pour vérifier
print(df[['ClientID', 'Nom']].head())


# In[3]:


# 2: Pseudonymisation des emails
df['Email'] = df['Email'].apply(lambda x: f"user_{df.index[df['Email'] == x].tolist()[0]}@example.com")

# Affichage du résultat pour vérifier
print(df[['ClientID', 'Email']].head())


# In[7]:





# In[9]:




# In[11]:


from faker import Faker

fake = Faker()

# 3: Anonymisation des numéros de téléphone
df['Téléphone'] = df['Téléphone'].apply(lambda x: fake.phone_number())

# Affichage du résultat pour vérifier
print(df[['ClientID', 'Téléphone']].head())


# In[12]:


# 4 & 5 Anonymisation des emails et des adresses

# 4: Anonymisation des emails
df['Email'] = df['Email'].apply(lambda x: fake.email())

# 5: Anonymisation des adresses
df['Adresse'] = df['Adresse'].apply(lambda x: fake.address())

# Affichage des 5 premières lignes pour vérifier
print(df[['ClientID', 'Email', 'Adresse']].head())


# In[16]:


from datetime import datetime
# 6:  Agrégation des âges

#  Normalisation du format de la date de naissance
df['DateNaissance'] = pd.to_datetime(df['DateNaissance'], errors='coerce')

# Suppression des valeurs non convertibles
df = df.dropna(subset=['DateNaissance'])

# Calcul de l'âge
current_year = datetime.now().year
df['age'] = df['DateNaissance'].apply(lambda x: current_year - x.year)

# Définition des tranches d'âge
bins = [0, 18, 30, 40, 50, 60, 70, 100]
labels = ["<18", "18-30", "30-40", "40-50", "50-60", "60-70", "70+"]
df['TrancheAge'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

# Vérification du résultat
print(df[['DateNaissance', 'age', 'TrancheAge']].head())


# In[17]:


# 7: Identification et correction des valeurs manquantes 
# a: Identification des valeurs manquantes 
missing_values = df.isnull().sum()
print("Valeurs manquantes par colonne :\n", missing_values[missing_values > 0])


# In[18]:


# Correction des valeurs manquantes
# Pour les colonnes numeriques(Remplacement par la médiane)
num_cols = df.select_dtypes(include=['number']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Pour les colonnes catégorielles (Remplacement par la modalité la plus fréquente)
cat_cols = df.select_dtypes(include=['object', 'bool']).columns
df[cat_cols] = df[cat_cols].apply(lambda x: x.fillna(x.mode()[0]))

# Vérification après traitement
print("\nValeurs manquantes après traitement :\n", df.isnull().sum().sum())


# In[20]:


# 8: Supression des doublons
df.drop_duplicates(inplace=True)
print("Nombre de doublons après suppression :", df.duplicated().sum())


# In[21]:


# 9: Suppression des valeurs aberrantes
# Identification des colonnes susceptibles d'avoir des anomalies
print(df[['FréquenceAchatMensuel', 'PanierMoyen', 'MontantTotalRemboursé']].describe())


# In[22]:


# Détection et suppression des valeurs aberrantes avec l'IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtrer les valeurs dans l'intervalle acceptable
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Suppression des valeurs aberrantes sur PanierMoyen et MontantTotalRemboursé
df_cleaned = remove_outliers(df, 'PanierMoyen')
df_cleaned = remove_outliers(df_cleaned, 'MontantTotalRemboursé')

# Vérification après suppression
print(df_cleaned.describe())


# In[23]:

print(df.columns)  

# 10: Ajout de nouvelles colonnes dérivées (Montant Total Dépensé & Client Fidèle)
df["MontantTotalDépensé"] = df["MontantTotalAchats"] - df["MontantTotalRemboursé"]
df["ClientFidèle"] = (df["FréquenceAchatMensuel"] > 5).astype(int)

# Copier les données traitées dans df_cleaned
df_cleaned = df.copy()

# Vérifier la description avant suppression des valeurs négatives
print(df_cleaned[["MontantTotalDépensé", "ClientFidèle"]].describe())

# 11: Identification des anomalies (Montant négatif)
clients_anormaux = df_cleaned[df_cleaned["MontantTotalDépensé"] < 0]
print(f"Nombre d'anomalies avant suppression : {clients_anormaux.shape[0]}")
print(clients_anormaux[["MontantTotalAchats", "MontantTotalRemboursé", "NombreRemboursements", "MontantTotalDépensé"]].head())

# Suppression des lignes avec MontantTotalDépensé négatif
df_cleaned = df_cleaned[df_cleaned["MontantTotalDépensé"] >= 0]

# Vérification après suppression
print(f"Nombre d'anomalies restantes après suppression : {df_cleaned[df_cleaned['MontantTotalDépensé'] < 0].shape[0]}")

# Sauvegarde du fichier corrigé
df_cleaned.to_csv("/data/processed/dataset_projet_evaluation_cleaned.csv", index=False, mode='w')

# In[ ]:



