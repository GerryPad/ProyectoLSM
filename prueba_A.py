import os
import cv2
import mediapipe as mp
import pandas as pd
import math
# -------------------------------------------------
# 1. RUTA ESPECÍFICA DE PRUEBA (Solo la A de train)
# -------------------------------------------------
ruta_carpeta_a = "dataset/MSL-ABC/lsm-abc-A/train/A"
#ruta_carpeta_a = "A_fotos"
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

datos = []
etiquetas = []

print(f"Procesando imágenes de la carpeta: {ruta_carpeta_a}")

# -------------------------------------------------
# 2. PROCESAMIENTO DE LA CARPETA
# -------------------------------------------------
contador_imagenes = 0
if os.path.exists(ruta_carpeta_a):
    for archivo in os.listdir(ruta_carpeta_a):
        if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            ruta_imagen = os.path.join(ruta_carpeta_a, archivo)
            
            imagen = cv2.imread(ruta_imagen)
            if imagen is None:
                continue
                
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagen_rgb)
            
            resultado = detector.detect(imagen_mp)
            
            if resultado.hand_landmarks:
                mano = resultado.hand_landmarks[0]

                #Determinar si es mano izquierda o derecha
                tipo_mano = resultado.handedness[0][0].category_name #Devuelve left o right

                fila_landmarks = []
                landmarks_relativos = []

                muneca = mano[0]#sera nuesto punto de referencia
                x0 = muneca.x
                y0 = muneca.y
                z0 = muneca.z

                for punto in mano:
                    x_relativo = punto.x - x0

                    if tipo_mano == "Left":
                        x_relativo = -x_relativo

                    y_relativo = punto.y - y0
                    z_relativo = punto.z - z0

                    landmarks_relativos.append([x_relativo, y_relativo, z_relativo])

                distancia_maxima = 0
                for x, y, z in landmarks_relativos:

                    distancia = math.sqrt(x**2 + y**2 + z**2)

                    if distancia > distancia_maxima:
                        distancia_maxima = distancia
                
                
                for x, y, z in landmarks_relativos:
                    x_normalizado = x / distancia_maxima
                    y_normalizado = y / distancia_maxima
                    z_normalizado = z / distancia_maxima

                    fila_landmarks.extend([x_normalizado, y_normalizado, z_normalizado])

                datos.append(fila_landmarks)
                etiquetas.append("A") # Etiqueta fija para esta prueba
            contador_imagenes += 1
            # Imprime un aviso cada 50 imágenes procesadas para ver que sigue vivo
            if contador_imagenes % 50 == 0:
                print(f"Procesadas {contador_imagenes} imágenes...")
else:
    print("La ruta especificada no existe. Revisa el nombre de tus carpetas.")

detector.close()

# -------------------------------------------------
# 3. GUARDAR CSV DE PRUEBA
# -------------------------------------------------
if len(datos) > 0:
    columnas = []
    for i in range(21):
        columnas.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
    
    df = pd.DataFrame(datos, columns=columnas)
    df['etiqueta'] = etiquetas
    
    df.to_csv("train_letra_A.csv", index=False)
    print(f"¡Prueba exitosa! Se guardaron {len(datos)} registros de la letra A en 'test_letra_A.csv'.")
else:
    print("No se detectaron manos en las imágenes de esta carpeta.")