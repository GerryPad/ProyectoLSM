import os #Conectar con el OS (para manejo de archivos)
#Usamos el backend grafico X11 (para evitar que abra multiples ventanas emergentes)
os.environ["QT_QPA_PLATFORM"] = "xcb" 
import cv2 #Manejo de video
import mediapipe as mp #Deteccion de manos
import joblib #Cargar los modelos entrenados
import time #Control de retardos
import math
import numpy as np

TIEMPO_MOSTRAR_DINAMICA = 0   #Segundos para delay (para ver las letras dinamicas)
tiempo_inicio_pausa = 0

#Carga de modelos entrenados
print("Cargando modelos...")
modelo_estatico = joblib.load("modelo_lsm_estatico.pkl")        
modelo_dinamico = joblib.load("modelo_lsm_dinamico_movimiento.pkl") 

ruta_modelo = "hand_landmarker.task"
NUM_FOTOGRAMAS_OBJETIVO = 75

UMBRAL_MOVIMIENTO = 0.025  #Distancia minima de movimiento en muñeca para considerar letra dinamica
FRAMES_MINIMOS_SEGINTO = 15 #Frames minimos para validar letra dinamica
FRAMES_MAXIMOS_SEGINTO = 60 #Frames maximos por seña
FRAMES_REPOSO_PARA_CERRAR = 8 #Frames que debe pasar quieto para cambiar a estatico

#Configuramos MediaPipe con su configuracion base
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

#Configuramos el modo VIDEO para el rastreo de una mano a la vez
opciones = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=ruta_modelo),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)
detector = HandLandmarker.create_from_options(opciones)

#Camara web predeterminada con resolucion ligera (640x480)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("No se pudo abrir la camara.")
    exit()

#Ventana flotante  para interfaz
cv2.namedWindow("Interprete alfebto LSM Completo", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Interprete alfabeto LSM Completo", 1024, 768)

print("Interprete iniciado")
print("Presiona 'ESC' para salir")

#Variables de estado
timestamp = 0 #Necesaria para llevar el conteo de frames en milisegundos, en modo VIIDEO MediPipe cuenta por milisegundos
fotogramas_signo = [] #Buffer para guardar la seña cn movimiento        
contador_frames_quietos = 0 #Para determinar cuando la mano dejo de moverse

estado_sistema = "ESPERANDO"   # ESPERANDO = modo estatico, CAPTURANDO = modo dinamico
letra_predicha = "Esperando letra..."
color_texto = (0, 0, 255)      
muneca_anterior = None

#Muestrear o "crear" frames oara llegar al objetivo
def convertir_a_75_frames(fotogramas):
    if not fotogramas:
        return None

    secuencia_uniforme = []
    #Interpolacion
    if len(fotogramas) < NUM_FOTOGRAMAS_OBJETIVO:
        indices = np.linspace(0, len(fotogramas) - 1, NUM_FOTOGRAMAS_OBJETIVO)
        for idx in indices:
            idx_inf, idx_sup = int(np.floor(idx)), int(np.ceil(idx))
            if idx_inf == idx_sup:
                secuencia_uniforme.extend(fotogramas[idx_inf])
            else:
                peso = idx - idx_inf
                frame_interpolado = [
                    (1 - peso) * a + peso * b
                    for a, b in zip(fotogramas[idx_inf], fotogramas[idx_sup])
                ]
                secuencia_uniforme.extend(frame_interpolado)
    #Muestreo
    else:
        indices = np.linspace(0, len(fotogramas) - 1, NUM_FOTOGRAMAS_OBJETIVO, dtype=int)
        for idx in indices:
            secuencia_uniforme.extend(fotogramas[idx])
            
    return secuencia_uniforme

#Valida que existan sufientes frames, aplica la interpolacion y actualiza la interfaz
def ejecutar_prediccion_dinamica(fotogramas):
    global letra_predicha, color_texto
    if len(fotogramas) >= FRAMES_MINIMOS_SEGINTO:
        secuencia_cruda = [f["vector"] for f in fotogramas]
        secuencia_uniforme = convertir_a_75_frames(secuencia_cruda)

        #Cada letra dinamica debe tener 4950 caracteristicas
        if secuencia_uniforme and len(secuencia_uniforme) == 4950:
            prediccion = modelo_dinamico.predict([secuencia_uniforme])[0]
            letra_predicha = prediccion
            color_texto = (0, 255, 0) 
            print(f"[DINAMICO EXITO] Letra reconocida: {letra_predicha}")
        else:
            letra_predicha = "Error de dimensiones"
            color_texto = (0, 0, 255)
    else:
        print("[AVISO] Movimiento dinamico muy corto.")

#Bucle principal
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) #Aplicamos efecto espejo
    alto, ancho, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #MediaPipe exige formato RGB
    
    imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb) #Pasamos de formato de NumPy a algo legible por MediaPipe
    timestamp += int(1000 / 30) #Simulacion de paso de tiempo a 30 FPS
    resultado = detector.detect_for_video(imagen_mp, timestamp) #Enviamos la imagen con su marca de tiempo

    movimiento_frame = 0.0

    #Si detecta la mano, toma las coordenadas de la muñeca y calcula distancia respecto a frame anterior
    if resultado.hand_landmarks:
        mano = resultado.hand_landmarks[0]
        tipo_mano = resultado.handedness[0][0].category_name
        muneca = mano[0]
        x0, y0, z0 = muneca.x, muneca.y, muneca.z

        if muneca_anterior is not None:
            dx_m = x0 - muneca_anterior[0]
            dy_m = y0 - muneca_anterior[1]
            dz_m = z0 - muneca_anterior[2]
            movimiento_frame = math.sqrt(dx_m**2 + dy_m**2 + dz_m**2)

        muneca_anterior = (x0, y0, z0)

        #Maquina de estados
        
        if estado_sistema == "ESPERANDO":
            #Procesamiento estatico
            fila_landmarks = []
            landmarks_relativos = []
            #Calculamos los landmarks relativos a la muñeca
            for punto in mano:
                x_relativo = punto.x - x0
                if tipo_mano == "Left":
                    x_relativo = -x_relativo
                y_relativo = punto.y - y0
                z_relativo = punto.z - z0
                landmarks_relativos.append([x_relativo, y_relativo, z_relativo])

            distancia_maxima = max(math.sqrt(x**2 + y**2 + z**2) for x, y, z in landmarks_relativos)

            #Normalizamos la escala, dividiendo por la distancia maxima            
            if distancia_maxima > 0:
                for x, y, z in landmarks_relativos:
                    fila_landmarks.extend([x / distancia_maxima, y / distancia_maxima, z / distancia_maxima])
                
                #Con el frame normalizado realizamos prediccion
                prediccion_estatica = modelo_estatico.predict([fila_landmarks])[0]
                letra_predicha = prediccion_estatica
                color_texto = (0, 255, 0)

            #Detectar movimientos para cambiar a modelo dinamico
            if movimiento_frame >= UMBRAL_MOVIMIENTO:
                estado_sistema = "CAPTURANDO"
                fotogramas_signo.clear()
                contador_frames_quietos = 0
                print("\n[INFO] Movimiento detectado -> Cambiando a Modo Dinamico...")

        #Modelo dinamico
        elif estado_sistema == "CAPTURANDO":
            #Almacenamos la forma normalizada de la mano
            if len(fotogramas_signo) == 0:
                xi, yi, zi = x0, y0, z0
            else:
                xi, yi, zi = fotogramas_signo[0]["origen"]

            landmarks_relativos = []
            for punto in mano:
                x_rel = punto.x - x0
                if tipo_mano == "Left":
                    x_rel = -x_rel
                landmarks_relativos.append([x_rel, punto.y - y0, punto.z - z0])

            dist_max = max(math.sqrt(x**2 + y**2 + z**2) for x, y, z in landmarks_relativos)
            
            if dist_max > 0:
                vector_f = []
                for x, y, z in landmarks_relativos:
                    vector_f.extend([x/dist_max, y/dist_max, z/dist_max])

                #Almacenamos el desplazamiento de la mano, respecto al origen
                dx_total = x0 - xi
                dy_total = y0 - yi
                dz_total = z0 - zi
                if tipo_mano == "Left":
                    dx_total = -dx_total

                vector_f.extend([dx_total, dy_total, dz_total])
                
                fotogramas_signo.append({
                    "vector": vector_f,
                    "origen": (xi, yi, zi)
                })

            if movimiento_frame < UMBRAL_MOVIMIENTO:
                contador_frames_quietos += 1
            else:
                contador_frames_quietos = 0 

            #Si la mano se mantiene quieta por determinado tiempo o se alcanza el maximo de frames se procesa la señal dinamica
            if (contador_frames_quietos >= FRAMES_REPOSO_PARA_CERRAR) or (len(fotogramas_signo) >= FRAMES_MAXIMOS_SEGINTO):
                estado_sistema = "PROCESANDO"
                print(f"[INFO] Procesando seña dinamica (Frames: {len(fotogramas_signo)})...")
                
                ejecutar_prediccion_dinamica(fotogramas_signo)
                
                fotogramas_signo.clear()

                #Mantener la letra dinamica por unos segundos
                estado_sistema = "PAUSA"
                tiempo_inicio_pausa = time.time()
                contador_frames_quietos = 0

        elif estado_sistema == "PAUSA":
            tiempo_transcurrido = time.time() - tiempo_inicio_pausa

            if tiempo_transcurrido >= TIEMPO_MOSTRAR_DINAMICA:
                estado_sistema = "ESPERANDO"
                muneca_anterior = (x0, y0, z0)
                print("[INFO] Pausa terminada -> Listo para nueva seña.")

        #Dibujar landmarks
        for punto in mano:
            cv2.circle(frame, (int(punto.x * ancho), int(punto.y * alto)), 4, (255, 0, 0), -1)

    #En caso de que se retire la mano de la camara
    else:
        muneca_anterior = None
        if estado_sistema == "CAPTURANDO" and len(fotogramas_signo) >= FRAMES_MINIMOS_SEGINTO:
            print("[INFO] Mano retirada, procesando seña dinámica...")
            ejecutar_prediccion_dinamica(fotogramas_signo)
        
        fotogramas_signo.clear()
        estado_sistema = "ESPERANDO"
        contador_frames_quietos = 0
        letra_predicha = "Mano no detectada"
        color_texto = (0, 0, 255)

    #Interfaz
    cv2.rectangle(frame, (20, 20), (550, 135), (50, 50, 50), -1)
    texto_pantalla = letra_predicha
    if str(letra_predicha).upper() == 'Ñ':
        texto_pantalla = "N~ (Eny)"
    
    cv2.putText(frame, f"Letra: {texto_pantalla}", (35, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, color_texto, 2)
    cv2.putText(frame, f"Modo: {estado_sistema}", (35, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Buffer: {len(fotogramas_signo)}", (35, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("Interprete LSM Completo", frame)

    if cv2.waitKey(1) & 0xFF == 27:  #ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
print("Interprete LSM finalizado.")