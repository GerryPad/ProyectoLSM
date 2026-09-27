import os
import cv2
import mediapipe as mp
import pandas as pd
import math
import numpy as np

# -------------------------------------------------
# 1. CONFIGURACIÓN DE RUTAS Y MEDIAPIPE
# -------------------------------------------------
carpeta_videos = "MSL-dynamic-signs/train" 
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

# Definimos el número de fotogramas objetivo global (ej. 60)
NUM_FOTOGRAMAS_OBJETIVO = 60
datos_videos = []
etiquetas = []

print(f"Procesando videos de la carpeta: {carpeta_videos}")

# -------------------------------------------------
# 2. PROCESAMIENTO MASIVO CON PADDING / ESTIRAMIENTO
# -------------------------------------------------
contador_videos = 0
videos_procesados = 0
videos_estirados = 0

if os.path.exists(carpeta_videos):
    for archivo in os.listdir(carpeta_videos):
        if archivo.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            ruta_video = os.path.join(carpeta_videos, archivo)
            
            # Extraer la letra del nombre del archivo (ej. 'S1-J-frontal-1.mp4' -> 'J')
            partes = archivo.split('-')
            if len(partes) >= 2:
                letra = partes[1].upper()
            else:
                continue
                
            cap = cv2.VideoCapture(ruta_video)
            fotogramas_video = []
            
            # Creamos un detector independiente para cada video
            detector = HandLandmarker.create_from_options(opciones)
            
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
            detector.close()
            
            # Validamos que el video tenga al menos al menos un fotograma con mano
            if len(fotogramas_video) > 0:
                secuencia_uniforme = []
                
                # SI EL VIDEO ES MÁS CORTO QUE EL OBJETIVO: Lo estiramos (interpolamos)
                if len(fotogramas_video) < NUM_FOTOGRAMAS_OBJETIVO:
                    videos_estirados += 1
                    indices = np.linspace(0, len(fotogramas_video) - 1, NUM_FOTOGRAMAS_OBJETIVO)
                    
                    for idx in indices:
                        idx_inf = int(np.floor(idx))
                        idx_sup = int(np.ceil(idx))
                        
                        if idx_inf == idx_sup:
                            secuencia_uniforme.extend(fotogramas_video[idx_inf])
                        else:
                            # Interpolación lineal entre el fotograma inferior y superior
                            peso = idx - idx_inf
                            frame_interpolado = [
                                (1 - peso) * a + peso * b 
                                for a, b in zip(fotogramas_video[idx_inf], fotogramas_video[idx_sup])
                            ]
                            secuencia_uniforme.extend(frame_interpolado)
                
                # SI EL VIDEO CUMPLE O SUPERA EL OBJETIVO: Aplicamos muestreo normal (downsampling)
                else:
                    indices = np.linspace(0, len(fotogramas_video) - 1, NUM_FOTOGRAMAS_OBJETIVO, dtype=int)
                    for idx in indices:
                        secuencia_uniforme.extend(fotogramas_video[idx])
                
                datos_videos.append(secuencia_uniforme)
                etiquetas.append(letra)
                videos_procesados += 1
            
            contador_videos += 1
            if contador_videos % 10 == 0:
                print(f"Procesados {contador_videos} videos...")
else:
    print("La ruta especificada no existe. Revisa el nombre de tus carpetas.")

# -------------------------------------------------
# 3. GUARDAR EL CSV DINÁMICO
# -------------------------------------------------
if len(datos_videos) > 0:
    columnas = []
    for f in range(NUM_FOTOGRAMAS_OBJETIVO):
        for i in range(21):
            columnas.extend([f'f{f}_x{i}', f'f{f}_y{i}', f'f{f}_z{i}'])
            
    df_dinamico = pd.DataFrame(datos_videos, columns=columnas)
    df_dinamico['etiqueta'] = etiquetas
    
    archivo_salida = "dataset_landmarks_dinamicos.csv"
    df_dinamico.to_csv(archivo_salida, index=False)
    print(f"\n¡Proceso dinámico finalizado!")
    print(f"Total de videos guardados exitosamente: {videos_procesados}")
    print(f"Videos cortos que fueron estirados automáticamente: {videos_estirados}")
    print(f"Archivo generado: '{archivo_salida}'")
else:
    print("No se pudieron procesar videos válidos. Revisa la ruta de la carpeta.")