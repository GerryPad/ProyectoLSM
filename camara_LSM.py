import cv2
import mediapipe as mp
import joblib
import math

# -------------------------------------------------
# 1. CARGAR EL MODELO ENTRENADO Y CONFIGURAR MEDIAPIPE
# -------------------------------------------------
print("Cargando el modelo de Machine Learning...")
modelo = joblib.load("modelo_lsm_estatico.pkl")

ruta_modelo = "hand_landmarker.task"
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

opciones = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=ruta_modelo),
    running_mode=RunningMode.VIDEO,
    num_hands=1 # Detectamos una mano a la vez para la predicción
)

detector = HandLandmarker.create_from_options(opciones)

# -------------------------------------------------
# 2. INICIALIZAR Y OPTIMIZAR LA CÁMARA WEB
# -------------------------------------------------
cap = cv2.VideoCapture(0)

# Forzamos una resolución más ligera para evitar el retraso (delay)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

print("\n¡Cámara optimizada iniciada! Realiza una seña frente a la lente.")
print("Presiona la tecla 'ESC' en tu teclado para salir.")

timestamp = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se pudo recibir fotograma de la cámara. Saliendo...")
        break

    # Volteamos el fotograma horizontalmente para que actúe como espejo (más natural para el usuario)
    frame = cv2.flip(frame, 1)
    
    # Convertir BGR a RGB para MediaPipe
    alto, ancho, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp+=int(1000/30)
    resultado = detector.detect_for_video(imagen_mp, timestamp)

    letra_predicha = "Esperando letra..."
    color_texto = (0, 0, 255) # Rojo por defecto

    # -------------------------------------------------
    # 3. PROCESAMIENTO Y PREDICCIÓN EN TIEMPO REAL
    # -------------------------------------------------
    if resultado.hand_landmarks:
        mano = resultado.hand_landmarks[0]
        
        # Obtenemos si es mano izquierda o derecha para aplicar el mismo filtro ambidextro
        tipo_mano = resultado.handedness[0][0].category_name
        
        fila_landmarks = []
        landmarks_relativos = []

        # Punto de referencia: La muñeca (landmark 0)
        muneca = mano[0]
        x0, y0, z0 = muneca.x, muneca.y, muneca.z

        for punto in mano:
            x_relativo = punto.x - x0
            
            # Efecto espejo matemático si detecta mano izquierda
            if tipo_mano == "Left":
                x_relativo = -x_relativo
                
            y_relativo = punto.y - y0
            z_relativo = punto.z - z0

            landmarks_relativos.append([x_relativo, y_relativo, z_relativo])

        # Normalización de escala (distancia máxima desde la muñeca)
        distancia_maxima = 0
        for x, y, z in landmarks_relativos:
            distancia = math.sqrt(x**2 + y**2 + z**2)
            if distancia > distancia_maxima:
                distancia_maxima = distancia
        
        # Evitamos división por cero por seguridad
        if distancia_maxima > 0:
            for x, y, z in landmarks_relativos:
                fila_landmarks.extend([
                    x / distancia_maxima, 
                    y / distancia_maxima, 
                    z / distancia_maxima
                ])

            # -------------------------------------------------
            # 4. PREDICCIÓN CON EL MODELO ENTRENADO (.pkl)
            # -------------------------------------------------
            # El modelo espera una matriz 2D, por eso envolvemos la lista en doble corchete
            prediccion = modelo.predict([fila_landmarks])
            letra_predicha = prediccion[0]
            color_texto = (0, 255, 0) # Verde cuando detecta y predice con éxito

        # Dibujar los 21 puntos clave y las conexiones básicas sobre la mano en pantalla
        for punto in mano:
            x_pixel = int(punto.x * ancho)
            y_pixel = int(punto.y * alto)
            cv2.circle(frame, (x_pixel, y_pixel), 4, (255, 0, 0), -1)

    # -------------------------------------------------
    # 5. INTERFAZ VISUAL EN LA VENTANA DE LA CÁMARA
    # -------------------------------------------------
    # Recuadro de fondo para el texto de la predicción
    cv2.rectangle(frame, (20, 20), (350, 100), (50, 50, 50), -1)
    
    # Texto de la letra detectada en grande
    cv2.putText(
        frame, 
        f"Letra: {letra_predicha}", 
        (35, 80), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1.5, 
        color_texto, 
        3
    )

    # Mostrar la ventana en tiempo real
    cv2.imshow("Reconocimiento de LSM en Vivo", frame)

    # Romper el ciclo si se presiona la tecla ESC (código 27)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Liberar la cámara y cerrar ventanas al terminar
cap.release()
cv2.destroyAllWindows()
detector.close()
print("Sesión de cámara finalizada.")