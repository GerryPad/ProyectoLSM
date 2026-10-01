import os   #para trabajar con archivos y carpetas
import cv2  #OpenCV, para leer y modificar imagenes
import mediapipe as mp  #MediaPipe, para detectar la mano y generar landmarks
import pandas as pd
import math
# Ruta del la carpeta donde estan las imagenes de la letra
ruta_carpeta_a = "dataset/MSL-ABC/lsm-abc-C/train/Y"
ruta_modelo = "hand_landmarker.task" #modelo de MediaPipe encargado de dectar las manos

#configuracion de MediaPipe(nombres mas cortos)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

#Configuracion del detector de mano
opciones = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=ruta_modelo), #indicar que modelo usar
    running_mode=RunningMode.IMAGE, #decimos que procesaremos imagenes independientes
    num_hands=1 #numero de manos a detectar
)
#creamos el detector
detector = HandLandmarker.create_from_options(opciones)

datos = []
etiquetas = []

print(f"Procesando imágenes de la carpeta: {ruta_carpeta_a}")

# Procesamos la carpeta
contador_imagenes = 0
if os.path.exists(ruta_carpeta_a): #ver si existe la carpeta
    for archivo in os.listdir(ruta_carpeta_a): #recorre todas las imagnes de la carpeta
        if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            ruta_imagen = os.path.join(ruta_carpeta_a, archivo)
            
            imagen = cv2.imread(ruta_imagen) #se carga la imagen
            if imagen is None: #si no se puede leer la ignora
                continue
                
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB) #convertimos a RGB para que MediaPipe pueda leerlo
            imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagen_rgb) #convertimo s formato de MediaPipe
            
            resultado = detector.detect(imagen_mp)  #detecta la mano

            #encontro una mano
            if resultado.hand_landmarks:
                mano = resultado.hand_landmarks[0] #contiene 21 landmarks

                #Determinar si es mano izquierda o derecha
                tipo_mano = resultado.handedness[0][0].category_name #Devuelve left o right

                fila_landmarks = []
                landmarks_relativos = [] 

                muneca = mano[0]#sera nuesto punto de referencia
                x0 = muneca.x
                y0 = muneca.y
                z0 = muneca.z

                #normalizacion de posicion
                for punto in mano:
                    x_relativo = punto.x - x0

                    if tipo_mano == "Left":
                        x_relativo = -x_relativo

                    y_relativo = punto.y - y0
                    z_relativo = punto.z - z0

                    landmarks_relativos.append([x_relativo, y_relativo, z_relativo])

                #normalizacion de escala
                distancia_maxima = 0
                for x, y, z in landmarks_relativos:

                    distancia = math.sqrt(x**2 + y**2 + z**2)

                    if distancia > distancia_maxima:
                        distancia_maxima = distancia
                
                for x, y, z in landmarks_relativos:
                    x_normalizado = x / distancia_maxima
                    y_normalizado = y / distancia_maxima
                    z_normalizado = z / distancia_maxima

                    fila_landmarks.extend([x_normalizado, y_normalizado, z_normalizado]) #usamos extend para hacer una lista de 63 numeros

                datos.append(fila_landmarks)
                etiquetas.append("Y") # Etiqueta fija, se modifica de acuerdo a la carpeta de la letra que se procese

            contador_imagenes += 1
            # Imprime un aviso cada 50 imágenes procesadas para ver que sigue vivo
            if contador_imagenes % 50 == 0:
                print(f"Procesadas {contador_imagenes} imágenes...")
else:
    print("La ruta especificada no existe. Revisa el nombre de tus carpetas.")

detector.close()

# Guardamos el CSV
if len(datos) > 0:
    columnas = []
    for i in range(21):
        columnas.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
    
    df = pd.DataFrame(datos, columns=columnas)
    df['etiqueta'] = etiquetas
    
    df.to_csv("train_letra_Y.csv", index=False)
    print(f"¡Prueba exitosa! Se guardaron {len(datos)} registros de la letra A en 'train_letra_Y.csv'.")
else:
    print("No se detectaron manos en las imágenes de esta carpeta.")