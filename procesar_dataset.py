import os
import cv2
import mediapipe as mp
import pandas as pd

# Inicializamos MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, # True porque procesamos imágenes fijas, no video en tiempo real
    max_num_hands=1,
    min_detection_confidence=0.5
)

# REEMPLAZA ESTA RUTA con la carpeta real donde descomprimiste el dataset
# Apunta directamente a la carpeta que contiene las letras A, B, C...
dataset_path = "MSL-ABC/lsm-abc-A/test"

datos = []
etiquetas = []

print("Procesando imágenes del dataset...")

# Recorremos cada carpeta (que representa una letra)
for letra in os.listdir(dataset_path):
    carpeta_letra = os.path.join(dataset_path, letra)
    
    # Nos aseguramos de que sea una carpeta real y evitamos archivos ocultos del sistema
    if os.path.isdir(carpeta_letra) and not letra.startswith('.'):
        print(f"-> Entrando a la carpeta: {letra}")
        
        for archivo_imagen in os.listdir(carpeta_letra):
            # Filtramos solo extensiones de imagen comunes
            if archivo_imagen.lower().endswith(('.png', '.jpg', '.jpeg')):
                ruta_imagen = os.path.join(carpeta_letra, archivo_imagen)
                
                img = cv2.imread(ruta_imagen)
                if img is None:
                    continue
                    
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                resultados = hands.process(img_rgb)
                
                if resultados.multi_hand_landmarks:
                    for hand_landmarks in resultados.multi_hand_landmarks:
                        fila_landmarks = []
                        for landmark in hand_landmarks.landmark:
                            fila_landmarks.extend([landmark.x, landmark.y, landmark.z])
                        
                        datos.append(fila_landmarks)
                        etiquetas.append(letra)

print(f"Total de registros extraídos: {len(datos)}")

# Convertimos los datos a un DataFrame de Pandas
# Tendremos 63 columnas de coordenadas (21 puntos * 3 ejes) + 1 columna de la etiqueta
columnas = []
for i in range(21):
    columnas.extend([f'x_{i}', f'y_{i}', f'z_{i}'])

df = pd.DataFrame(datos, columns=columnas)
df['etiqueta'] = etiquetas

# Guardamos el resultado en un archivo CSV listo para entrenar
df.to_csv("dataset_landmarks_estaticos.csv", index=False)
print("¡Proceso finalizado! Archivo guardado como 'dataset_landmarks_estaticos.csv'.")