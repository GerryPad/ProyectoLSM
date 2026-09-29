import cv2
import mediapipe as mp
import joblib
import math
import numpy as np
import time

# -------------------------------------------------
# 1. CONFIGURACIÓN
# -------------------------------------------------

modelo = joblib.load("modelo_lsm_dinamico.pkl")
ruta_modelo = "hand_landmarker.task"

NUM_FOTOGRAMAS_OBJETIVO = 75
DURACION_GRABACION = 2.5   # segundos

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
# 2. CÁMARA
# -------------------------------------------------

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

print("Presiona ESPACIO para grabar una seña dinámica.")
print("Presiona ESC para salir.")

timestamp = 0
grabando = False
inicio_grabacion = 0

fotogramas_secuencia = []

letra_predicha = "Esperando..."
color_texto = (0, 0, 255)

# -------------------------------------------------
# 3. BUCLE PRINCIPAL
# -------------------------------------------------

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    alto, ancho, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
    # 4. OBTENER LANDMARKS
    # -------------------------------------------------

    if resultado.hand_landmarks:

        mano = resultado.hand_landmarks[0]

        tipo_mano = resultado.handedness[0][0].category_name

        landmarks_relativos = []

        muneca = mano[0]

        x0 = muneca.x
        y0 = muneca.y
        z0 = muneca.z

        for punto in mano:

            x_rel = punto.x - x0

            if tipo_mano == "Left":
                x_rel = -x_rel

            y_rel = punto.y - y0
            z_rel = punto.z - z0

            landmarks_relativos.append([
                x_rel,
                y_rel,
                z_rel
            ])

        # Normalización igual que procesar_videos.py

        dist_max = max(
            math.sqrt(x**2 + y**2 + z**2)
            for x, y, z in landmarks_relativos
        )

        if dist_max > 0:

            vector_fotograma = []

            for x, y, z in landmarks_relativos:

                vector_fotograma.extend([
                    x / dist_max,
                    y / dist_max,
                    z / dist_max
                ])

            # Solo guardamos los frames cuando estamos grabando

            if grabando:
                fotogramas_secuencia.append(vector_fotograma)

        # Dibujar puntos de la mano

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

    # -------------------------------------------------
    # 5. TERMINAR GRABACIÓN
    # -------------------------------------------------

    if grabando:

        tiempo_transcurrido = time.time() - inicio_grabacion

        cv2.putText(
            frame,
            "GRABANDO...",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        if tiempo_transcurrido >= DURACION_GRABACION:

            grabando = False

            print(
                "Frames detectados:",
                len(fotogramas_secuencia)
            )

            # -----------------------------------------
            # 6. CONVERTIR LA SECUENCIA A 75 FRAMES
            # -----------------------------------------

            if len(fotogramas_secuencia) > 0:

                secuencia_uniforme = []

                # Si tenemos menos de 75 frames:
                # interpolamos exactamente como en entrenamiento

                if len(fotogramas_secuencia) < NUM_FOTOGRAMAS_OBJETIVO:

                    indices = np.linspace(
                        0,
                        len(fotogramas_secuencia) - 1,
                        NUM_FOTOGRAMAS_OBJETIVO
                    )

                    for idx in indices:

                        idx_inf = int(np.floor(idx))
                        idx_sup = int(np.ceil(idx))

                        if idx_inf == idx_sup:

                            secuencia_uniforme.extend(
                                fotogramas_secuencia[idx_inf]
                            )

                        else:

                            peso = idx - idx_inf

                            frame_interpolado = [
                                (1 - peso) * a + peso * b
                                for a, b in zip(
                                    fotogramas_secuencia[idx_inf],
                                    fotogramas_secuencia[idx_sup]
                                )
                            ]

                            secuencia_uniforme.extend(
                                frame_interpolado
                            )

                # Si tenemos 75 o más:
                # hacemos downsampling
                else:

                    indices = np.linspace(
                        0,
                        len(fotogramas_secuencia) - 1,
                        NUM_FOTOGRAMAS_OBJETIVO,
                        dtype=int
                    )

                    for idx in indices:

                        secuencia_uniforme.extend(
                            fotogramas_secuencia[idx]
                        )

                # -----------------------------------------
                # 7. PREDICCIÓN
                # -----------------------------------------

                print(
                    "Características:",
                    len(secuencia_uniforme)
                )

                prediccion = modelo.predict(
                    [secuencia_uniforme]
                )

                letra_predicha = prediccion[0]

                color_texto = (0, 255, 0)

                print(
                    "Letra dinámica detectada:",
                    letra_predicha
                )

            else:

                letra_predicha = "No se detecto mano"
                color_texto = (0, 0, 255)

    # -------------------------------------------------
    # 8. MOSTRAR RESULTADO
    # -------------------------------------------------

    cv2.rectangle(
        frame,
        (20, 20),
        (420, 100),
        (50, 50, 50),
        -1
    )

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
        "ESPACIO = grabar | ESC = salir",
        (20, alto - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Reconocimiento LSM Dinamico",
        frame
    )

    # -------------------------------------------------
    # 9. TECLADO
    # -------------------------------------------------

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == 27:
        break

    if tecla == 32 and not grabando:

        print("\nGrabando nueva seña...")

        fotogramas_secuencia = []

        letra_predicha = "Grabando..."

        color_texto = (0, 255, 255)

        inicio_grabacion = time.time()

        grabando = True


# -------------------------------------------------
# 10. CERRAR
# -------------------------------------------------

cap.release()
cv2.destroyAllWindows()
detector.close()

print("Programa finalizado.")