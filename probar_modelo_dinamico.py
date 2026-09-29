import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score
)

# -------------------------------------------------
# 1. CARGAR DATASET Y MODELO
# -------------------------------------------------

archivo_dataset = "dataset_landmarks_dinamicos.csv"
archivo_modelo = "modelo_lsm_dinamico.pkl"

print("Cargando dataset...")
df = pd.read_csv(archivo_dataset)

print("Cargando modelo...")
modelo = joblib.load(archivo_modelo)

# Separamos características y etiquetas
X = df.drop(columns=["etiqueta"])
y = df["etiqueta"]

print("\nTotal de muestras:", len(df))
print("Número de características:", X.shape[1])

print("\nCantidad de videos por letra:")
print(y.value_counts().sort_index())


# -------------------------------------------------
# 2. DIVIDIR DATASET
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nMuestras de entrenamiento:", len(X_train))
print("Muestras de prueba:", len(X_test))


# -------------------------------------------------
# 3. HACER PREDICCIONES
# -------------------------------------------------

y_pred = modelo.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n====================================")
print("RESULTADOS GENERALES")
print("====================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nReporte de clasificación:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# -------------------------------------------------
# 4. ANALIZAR ESPECÍFICAMENTE Q, K Y X
# -------------------------------------------------

letras_problematicas = ["Q", "K", "X"]

print("\n====================================")
print("ANÁLISIS DE Q, K Y X")
print("====================================")

for letra in letras_problematicas:

    print(f"\n----------- LETRA {letra} -----------")

    # Seleccionamos únicamente ejemplos cuya letra real sea esta
    mascara = y_test == letra

    reales = y_test[mascara]
    predicciones = y_pred[mascara]

    total = len(reales)

    if total == 0:
        print("No existen ejemplos de esta letra en test.")
        continue

    correctas = sum(predicciones == letra)

    print("Ejemplos evaluados:", total)
    print("Predicciones correctas:", correctas)

    print(
        f"Precisión para {letra}: "
        f"{correctas / total * 100:.2f}%"
    )

    print("\nEl modelo respondió:")

    conteo = pd.Series(predicciones).value_counts()

    for prediccion, cantidad in conteo.items():

        porcentaje = cantidad / total * 100

        print(
            f"   {prediccion}: "
            f"{cantidad} veces "
            f"({porcentaje:.2f}%)"
        )


# -------------------------------------------------
# 5. MOSTRAR TODOS LOS ERRORES Q/K/X
# -------------------------------------------------

print("\n====================================")
print("ERRORES EN Q, K Y X")
print("====================================")

for real, predicha in zip(y_test, y_pred):

    if real in letras_problematicas and real != predicha:

        print(
            f"Real: {real}  --->  "
            f"Predicción: {predicha}"
        )


# -------------------------------------------------
# 6. MATRIZ DE CONFUSIÓN
# -------------------------------------------------

etiquetas = sorted(y.unique())

matriz = confusion_matrix(
    y_test,
    y_pred,
    labels=etiquetas
)

fig, ax = plt.subplots(figsize=(10, 8))

display = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=etiquetas
)

display.plot(
    ax=ax,
    xticks_rotation=45,
    cmap="Blues",
    values_format="d"
)

plt.title("Matriz de confusión - Modelo LSM dinámico")
plt.tight_layout()
plt.show()