import os
import cv2
import numpy as np

carpeta_videos = "MSL-dynamic-signs/train" # Ajusta tu ruta si es diferente
conteo_frames = []

print("Analizando la duración de los videos dinámicos...")

if os.path.exists(carpeta_videos):
    for archivo in os.listdir(carpeta_videos):
        if archivo.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            ruta_video = os.path.join(carpeta_videos, archivo)
            cap = cv2.VideoCapture(ruta_video)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                conteo_frames.append(total_frames)
            cap.release()
            
    if len(conteo_frames) > 0:
        print(f"\n--- Resultados del Análisis ---")
        print(f"Total de videos analizados: {len(conteo_frames)}")
        print(f"Promedio de fotogramas por video: {np.mean(conteo_frames):.1f}")
        print(f"Video más corto: {min(conteo_frames)} fotogramas")
        print(f"Video más largo: {max(conteo_frames)} fotogramas")
    else:
        print("No se encontraron videos válidos para analizar.")
else:
    print(f"La ruta '{carpeta_videos}' no existe.")