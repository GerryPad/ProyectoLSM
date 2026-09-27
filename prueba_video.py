import os
import cv2
import mediapipe as mp
import pandas as pd
import math
import numpy as np

# -------------------------------------------------
# 1. RUTA DE UN SOLO VIDEO DE PRUEBA
# -------------------------------------------------
# Cambia esta ruta por la de un archivo de video real que tengas en tu carpeta
ruta_video_prueba = "MSL-dynamic-signs/train/S1-J-frontal-1.mp4" 
ruta_modelo = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

opciones = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=ruta_modelo),
    running_mode=RunningMode.IMAGE,
    num_hands=1
)

detector = HandLandmarker.create_from_options(opciones)

# Puedes cambiar este número de fotogramas objetivo para probar (ej. 60 u 80)
NUM_FOTOGRAMAS_OBJETIVO = 60

print(f"Procesando video individual: {ruta_video_prueba}")

cap = cv2.VideoCapture(ruta_video_prueba)
fotogramas_video = []

if not cap.isOpened():
    print(f"Error: No se pudo abrir el video en la ruta: {ruta_video_prueba}")
else:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        resultado = detector.detect(imagen_mp)
        
        if resultado.hand_landmarks:
            mano = resultado.hand_landmarks[0]
            tipo_mano = resultado.handedness[0][0].category_name
            
            landmarks_relativos = []
            muneca = mano[0]
            x0, y0, z0 = muneca.x, muneca.y, muneca.z
            
            for punto in mano:
                x_rel = punto.x - x0
                if tipo_mano == "Left":
                    x_rel = -x_rel  # Efecto espejo ambidextro
                y_rel = punto.y - y0
                z_rel = punto.z - z0
                landmarks_relativos.append([x_rel, y_rel, z_rel])
                
            dist_max = max(math.sqrt(x**2 + y**2 + z**2) for x, y, z in landmarks_relativos)
            
            if dist_max > 0:
                vector_fotograma = []
                for x, y, z in landmarks_relativos:
                    vector_fotograma.extend([x / dist_max, y / dist_max, z / dist_max])
                fotogramas_video.append(vector_fotograma)
                
    cap.release()
    print(f"Fotogramas totales con mano detectada en este video: {len(fotogramas_video)}")

detector.close()

# -------------------------------------------------
# 2. APLICAR MUESTREO Y GUARDAR CSV DE PRUEBA
# -------------------------------------------------
if len(fotogramas_video) >= NUM_FOTOGRAMAS_OBJETIVO:
    indices = np.linspace(0, len(fotogramas_video) - 1, NUM_FOTOGRAMAS_OBJETIVO, dtype=int)
    secuencia_uniforme = []
    
    for idx in indices:
        secuencia_uniforme.extend(fotogramas_video[idx])
        
    # Construir las columnas para el CSV (ej. f0_x0, f0_y0, f0_z0 ... hasta el fotograma objetivo)
    columnas = []
    for f in range(NUM_FOTOGRAMAS_OBJETIVO):
        for i in range(21):
            columnas.extend([f'f{f}_x{i}', f'f{f}_y{i}', f'f{f}_z{i}'])
            
    df_prueba = pd.DataFrame([secuencia_uniforme], columns=columnas)
    
    # Extraer letra del nombre del archivo de prueba
    nombre_archivo = os.path.basename(ruta_video_prueba)
    letra = nombre_archivo.split('-')[1].upper() if '-' in nombre_archivo else 'DESCONOCIDA'
    df_prueba['etiqueta'] = letra
    
    archivo_salida = "test_video_dinamico.csv"
    df_prueba.to_csv(archivo_salida, index=False)
    print(f"¡Prueba unitaria exitosa! Se guardó la secuencia de la letra '{letra}' en '{archivo_salida}'.")
    print(f"Dimensiones del DataFrame resultante: {df_prueba.shape}")
else:
    print(f"El video tiene menos fotogramas detectados ({len(fotogramas_video)}) que el objetivo ({NUM_FOTOGRAMAS_OBJETIVO}).")