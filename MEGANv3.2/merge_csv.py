import pandas as pd

def merge_and_keep_duplicates(main_file, aux_file, ecotype_id, ecotype_id_lower, ecotype_id_upper):
    # Carica i file principale e ausiliario
    main_df = pd.read_csv(main_file)
    aux_df = pd.read_csv(aux_file)

    # Step 1: Seleziona le righe con Ecotype_ID specificato e aggiungile al file principale
    selected_rows = aux_df[aux_df['EcotypeID'] == ecotype_id]

    merged_df = pd.concat([main_df, selected_rows])
    merged_df.reset_index(drop=True, inplace=True) 

    # Step 2: Salva i grid_ID delle righe aggiunte in un array
    grid_ids_to_remove = selected_rows['gridID'].tolist()

    # Step 3: Rimuovi dal file originale le righe con gli stessi grid_ID
    # e con Ecotype_ID diverso da 5867, 5868 o 5869
    condition_to_remove = (
        (merged_df['gridID'].isin(grid_ids_to_remove)) &
        ((merged_df['EcotypeID'] < ecotype_id_lower) | (merged_df['EcotypeID'] > ecotype_id_upper))
    )

    merged_df = merged_df[~condition_to_remove]

    # Salva il risultato nel file principale
    merged_df.to_csv(main_file, index=False)

# Esempio di utilizzo
merge_and_keep_duplicates('main.csv', 'grid_ecotype_pine.csv', 5867, 5867, 5869)
merge_and_keep_duplicates('main.csv', 'grid_ecotype_spruce.csv', 5868, 5867, 5869)
merge_and_keep_duplicates('main.csv', 'grid_ecotype_birch.csv', 5869, 5867, 5869)

