import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2
import mediapipe as mp
import joblib
import math
import numpy as np
from collections import deque

# -------------------------------------------------
# 1. CONFIGURACIÓN
# -------------------------------------------------

modelo = joblib.load("modelo_lsm_dinamico_movimiento.pkl")
ruta_modelo = "hand_landmarker.task"

NUM_FOTOGRAMAS_OBJETIVO = 75

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

opciones = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=ruta_modelo),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

detector = HandLandmarker.create_from_options(opciones)

# -------------------------------------------------
# 2. CÁMARA Y ESTRUCTURAS DE VENTANA DESLIZANTE
# -------------------------------------------------
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

# ---> AGREGA ESTA LÍNEA PARA FIJAR LA VENTANA <---
cv2.namedWindow("Reconocimiento Dinamico Continuo", cv2.WINDOW_NORMAL)

# Forzamos un tamaño inicial grande (ej. 1024x768 o 1280x720)
cv2.resizeWindow("Reconocimiento Dinamico Continuo", 1024, 768)

print("¡Cámara en modo continuo iniciada! Realiza tus señas dinámicas frente a la lente.")
print("Presiona la tecla 'ESC' para salir.")

timestamp = 0

# Colas circulares para mantener exactamente los últimos 75 fotogramas en tiempo real
buffer_caracteristicas = deque(maxlen=NUM_FOTOGRAMAS_OBJETIVO)
buffer_munecas = deque(maxlen=NUM_FOTOGRAMAS_OBJETIVO)

letra_predicha = "Esperando seña..."
color_texto = (0, 0, 255)

# -------------------------------------------------
# 3. BUCLE PRINCIPAL EN TIEMPO REAL
# -------------------------------------------------

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    alto, ancho, _ = frame.shape

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    imagen_mp = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    timestamp += int(1000 / 30)

    resultado = detector.detect_for_video(
        imagen_mp,
        timestamp
    )

    # -------------------------------------------------
    # 4. OBTENER LANDMARKS Y TRAYECTORIA DINÁMICA
    # -------------------------------------------------

    if resultado.hand_landmarks:

        mano = resultado.hand_landmarks[0]
        tipo_mano = resultado.handedness[0][0].category_name

        muneca = mano[0]
        x0, y0, z0 = muneca.x, muneca.y, muneca.z

        # Guardamos la muñeca actual en el historial de la ventana
        buffer_munecas.append((x0, y0, z0, tipo_mano))

        # La referencia inicial (xi, yi, zi) para calcular el desplazamiento (dx, dy, dz)
        # es el fotograma más antiguo actualmente dentro de nuestra ventana deslizante
        xi, yi, zi, _ = buffer_munecas[0]

        dx = x0 - xi
        dy = y0 - yi
        dz = z0 - zi

        if tipo_mano == "Left":
            dx = -dx

        # Landmarks relativos a la muñeca actual del fotograma
        landmarks_relativos = []
        for punto in mano:
            x_rel = punto.x - x0
            if tipo_mano == "Left":
                x_rel = -x_rel
            y_rel = punto.y - y0
            z_rel = punto.z - z0
            landmarks_relativos.append([x_rel, y_rel, z_rel])

        # Normalización de escala
        dist_max = max(
            math.sqrt(x**2 + y**2 + z**2)
            for x, y, z in landmarks_relativos
        )

        if dist_max > 0:
            vector_fotograma = []

            # 63 características de forma normalizada
            for x, y, z in landmarks_relativos:
                vector_fotograma.extend([
                    x / dist_max,
                    y / dist_max,
                    z / dist_max
                ])

            # 3 características de trayectoria de muñeca (dx, dy, dz) -> Total: 66
            vector_fotograma.extend([dx, dy, dz])

            # Añadimos el fotograma procesado a la ventana deslizante
            buffer_caracteristicas.append(vector_fotograma)

        # -------------------------------------------------
        # 5. PREDICCIÓN AUTOMÁTICA CUANDO EL BUFFER SE LLENA
        # -------------------------------------------------
        if len(buffer_caracteristicas) == NUM_FOTOGRAMAS_OBJETIVO:
            
            # Aplanamos los 75 fotogramas de la ventana (75 * 66 = 4950 características)
            secuencia_entrada = []
            for f in buffer_caracteristicas:
                secuencia_entrada.extend(f)

            prediccion = modelo.predict([secuencia_entrada])
            letra_predicha = prediccion[0]
            color_texto = (0, 255, 0)
        else:
            # Nuevo: Feedback visual mientras se llena la ventana de 75 frames
            letra_predicha = f"Capturando ({len(buffer_caracteristicas)}/75)..."
            color_texto = (0, 255, 255) # Color amarillo

        # Dibujar puntos de la mano en pantalla
        for punto in mano:
            x_pixel = int(punto.x * ancho)
            y_pixel = int(punto.y * alto)
            cv2.circle(
                frame,
                (x_pixel, y_pixel),
                4,
                (255, 0, 0),
                -1
            )

    else:
        # Si la cámara pierde la mano temporalmente, vaciamos las colas para reiniciar el flujo
        buffer_caracteristicas.clear()
        buffer_munecas.clear()
        letra_predicha = "Mano no detectada..."
        color_texto = (0, 0, 255)

    # -------------------------------------------------
    # 6. INTERFAZ VISUAL EN TIEMPO REAL
    # -------------------------------------------------

    cv2.rectangle(frame, (20, 20), (420, 100), (50, 50, 50), -1)

    cv2.putText(
        frame,
        f"Letra: {letra_predicha}",
        (35, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        color_texto,
        3
    )

    cv2.putText(
        frame,
        "MODO EN VIVO | ESC = salir",
        (20, alto - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow("Reconocimiento Dinamico Continuo", frame)

    # -------------------------------------------------
    # 7. CONTROL DE SALIDA (TECLADO)
    # -------------------------------------------------

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == 27:  # ESC para salir
        break

# -------------------------------------------------
# 8. CERRAR RECURSOS
# -------------------------------------------------

cap.release()
cv2.destroyAllWindows()
detector.close()

print("Sesión de cámara en vivo finalizada.")