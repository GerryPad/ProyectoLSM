import os
import pandas as pd

# -------------------------------------------------
# 1. RUTA DE LA CARPETA CON LOS CSVs INDIVIDUALES
# -------------------------------------------------
carpeta_csvs = "LSM_Train_Completo"  # Ajusta el nombre si tu carpeta se llama distinto
archivo_salida = "dataset_landmarks_estaticos_completo.csv"

lista_dataframes = []

print("Buscando y combinando archivos CSV...")

# -------------------------------------------------
# 2. RECORRER Y LEER CADA CSV
# -------------------------------------------------
if os.path.exists(carpeta_csvs):
    for archivo in os.listdir(carpeta_csvs):
        if archivo.lower().endswith('.csv'):
            ruta_csv = os.path.join(carpeta_csvs, archivo)
            
            try:
                # Leemos el archivo CSV individual
                df_temporal = pd.read_csv(ruta_csv)
                lista_dataframes.append(df_temporal)
                print(f"-> Archivo integrado: {archivo} ({len(df_temporal)} registros)")
            except Exception as e:
                print(f"Error al leer {archivo}: {e}")
                
    # -------------------------------------------------
    # 3. UNIFICAR Y GUARDAR EL DATASET MAESTRO
    # -------------------------------------------------
    if len(lista_dataframes) > 0:
        # Concatenamos todos los dataframes en uno solo
        dataset_maestro = pd.concat(lista_dataframes, ignore_index=True)
        
        # Guardamos el archivo final combinado
        dataset_maestro.to_csv(archivo_salida, index=False)
        print(f"\n ¡Proceso exitoso!")
        print(f"Se unificaron {len(lista_dataframes)} archivos.")
        print(f"Total de registros combinados: {len(dataset_maestro)}")
        print(f"Archivo guardado como: '{archivo_salida}'")
    else:
        print("No se encontraron archivos CSV válidos en la carpeta.")
else:
    print(f"La carpeta '{carpeta_csvs}' no existe. Verifica la ruta.")