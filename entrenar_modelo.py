import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#Dataset completo de las letras(dinamicas o estaticas)
archivo_dataset = "dataset_landmarks_estaticos_completo.csv" 
#archivo_dataset = "dataset_landmarks_dinamico_movimiento.csv"
print(f"Cargando el dataset desde '{archivo_dataset}'...")

df = pd.read_csv(archivo_dataset)

# Separamos las características (coordenadas x, y, z de los 21 landmarks) de la etiqueta (letra)
X = df.drop(columns=['etiqueta'])
y = df['etiqueta']

print(f"Total de muestras cargadas: {len(df)}")

#Dividimos entre entramiento y prueba (80 y 20)
# Usamos 'stratify=y' para asegurar que todas las letras tengan la misma proporción en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Muestras para entrenamiento: {len(X_train)}")
print(f"Muestras para prueba: {len(X_test)}")

# Entrenamos el clasificador (Random Forest)
print("\nEntrenando el modelo de Random Forest...")
# 'n_jobs=-1' utiliza todos los núcleos de tu procesador para acelerar el entrenamiento
modelo = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
modelo.fit(X_train, y_train)
print("Entrenamiento finalizado ")

# Evaluamos el rendimiento del modelo
print("\nEvaluando el modelo con el conjunto de prueba...")
y_pred = modelo.predict(X_test)

precision_global = accuracy_score(y_test, y_pred)
print(f"\nPrecision global del modelo: {precision_global * 100:.2f}%\n")

# Reporte detallado por letra (precisión, recall, f1-score para cada seña)
print("Reporte de clasificación detallado por letra:")
print(classification_report(y_test, y_pred))

# Matriz de confucion
print("\nGenerando matriz de confusion...")

# Calcular la matriz
matriz_conf = confusion_matrix(y_test, y_pred, labels=modelo.classes_)

# Configurar la visualización gráfica
fig, ax = plt.subplots(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=matriz_conf, display_labels=modelo.classes_)

# Dibujarmo la matriz con colores legibles
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical', colorbar=True)
plt.title("Matriz de Confusion - Señas Estaticas LSM")
plt.tight_layout()

# Guardamos la imagen
nombre_imagen_matriz = "matriz_confusion_lsm_estaticas.png"
plt.savefig(nombre_imagen_matriz, dpi=300)
print(f"Matriz de confusión guardada como '{nombre_imagen_matriz}'")

# Mostrar la ventana gráfica opcionalmente
plt.show()

# Guardammos el modelo de entrenamiento
nombre_archivo_modelo = "modelo_lsm_estatico.pkl"
joblib.dump(modelo, nombre_archivo_modelo)
print(f"\nModelo guardado exitosamente como '{nombre_archivo_modelo}'.")