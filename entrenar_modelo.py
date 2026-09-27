import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# -------------------------------------------------
# 1. CARGAR EL DATASET MAESTRO COMBINADO
# -------------------------------------------------
archivo_dataset = "dataset_landmarks_estaticos_completo.csv" # Asegúrate de que coincida con el nombre de tu archivo
print(f"Cargando el dataset desde '{archivo_dataset}'...")

df = pd.read_csv(archivo_dataset)

# Separamos las características (coordenadas x, y, z de los 21 landmarks) de la etiqueta (letra)
X = df.drop(columns=['etiqueta'])
y = df['etiqueta']

print(f"Total de muestras cargadas: {len(df)}")

# -------------------------------------------------
# 2. DIVIDIR EN ENTRENAMIENTO Y PRUEBA (80/20)
# -------------------------------------------------
# Usamos 'stratify=y' para asegurar que todas las letras tengan la misma proporción en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Muestras para entrenamiento: {len(X_train)}")
print(f"Muestras para prueba: {len(X_test)}")

# -------------------------------------------------
# 3. ENTRENAR EL CLASIFICADOR (Random Forest)
# -------------------------------------------------
print("\nEntrenando el modelo de Random Forest (esto puede tomar unos momentos)...")
# 'n_jobs=-1' utiliza todos los núcleos de tu procesador para acelerar el entrenamiento
modelo = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
modelo.fit(X_train, y_train)
print("¡Entrenamiento finalizado con éxito!")

# -------------------------------------------------
# 4. EVALUAR EL RENDIMIENTO DEL MODELO
# -------------------------------------------------
print("\nEvaluando el modelo con el conjunto de prueba...")
y_pred = modelo.predict(X_test)

precision_global = accuracy_score(y_test, y_pred)
print(f"\nPrecisión global del modelo: {precision_global * 100:.2f}%\n")

# Reporte detallado por letra (precisión, recall, f1-score para cada seña)
print("Reporte de clasificación detallado por letra:")
print(classification_report(y_test, y_pred))

# -------------------------------------------------
# 5. GUARDAR EL MODELO ENTRENADO
# -------------------------------------------------
nombre_archivo_modelo = "modelo_lsm_estatico.pkl"
joblib.dump(modelo, nombre_archivo_modelo)
print(f"\nModelo guardado exitosamente como '{nombre_archivo_modelo}'.")