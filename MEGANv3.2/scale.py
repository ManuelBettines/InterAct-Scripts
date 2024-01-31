import pandas as pd

# Carica il file CSV
df = pd.read_csv('main.csv')

# Itera su tutti i gridID
for gridID in df['gridID'].unique():
    # Seleziona le righe corrispondenti al gridID corrente
    subset = df[df['gridID'] == gridID]

    # Calcola la somma degli Ecotype_Frac
    total_frac = subset['EcotypeFrac'].sum()

    # Se la somma è minore di uno, aggiusta le proporzioni
    if total_frac < 1:
        # Calcola il fattore di scala per riportare la somma a 1
        scale_factor = 1 / total_frac

        # Modifica le Ecotype_Frac
        df.loc[df['gridID'] == gridID, 'EcotypeFrac'] *= scale_factor

# Ora il DataFrame è aggiornato con le Ecotype_Frac corrette
# Puoi salvare il DataFrame modificato in un nuovo file CSV
df.to_csv('ecotype_main.csv', index=False)

