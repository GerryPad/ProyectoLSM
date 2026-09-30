import cv2
import mediapipe as mp
import joblib
import math
import numpy as np
import time
from collections import deque, Counter

# -------------------------------------------------
# 1. CONFIGURACIÓN
# -------------------------------------------------

modelo = joblib.load("modelo_lsm_dinamico_movimiento.pkl")
ruta_modelo = "hand_landmarker.task"

NUM_FOTOGRAMAS_OBJETIVO = 75

# Cantidad máxima de frames crudos que conservamos.
# A ~30 FPS, 60 frames son aproximadamente 2 segundos.
TAM_BUFFER = 60

# No intentamos reconocer hasta tener al menos estos frames.
MIN_FRAMES = 25

# Cada cuántos frames hacemos una predicción.
# No hace falta ejecutar Random Forest en absolutamente cada frame.
INTERVALO_PREDICCION = 5

# Umbral inicial de movimiento.
# ESTE VALOR ES DE PRUEBA. Hay que ajustarlo viendo la cámara.
UMBRAL_MOVIMIENTO = 0.04

# Para aceptar una letra pedimos varias predicciones recientes.
NUM_PREDICCIONES_ESTABLES = 4

# Número de predicciones que conservamos para votar.
TAM_HISTORIAL_PREDICCIONES = 5


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

print("Reconocimiento dinámico automático iniciado.")
print("Ya NO necesitas presionar ESPACIO.")
print("Presiona ESC para salir.")


# -------------------------------------------------
# 3. VARIABLES DEL RECONOCIMIENTO
# -------------------------------------------------

timestamp = 0

# Buffer continuo.
# Guardaremos landmarks y datos necesarios de los últimos frames.
buffer_frames = deque(maxlen=TAM_BUFFER)

# Historial de predicciones para estabilización.
historial_predicciones = deque(
    maxlen=TAM_HISTORIAL_PREDICCIONES
)

letra_predicha = "Esperando..."
color_texto = (0, 0, 255)

contador_frames = 0

movimiento_actual = 0.0


# -------------------------------------------------
# 4. FUNCIÓN PARA CONVERTIR LA VENTANA A 75 FRAMES
# -------------------------------------------------

def convertir_a_75_frames(fotogramas):

    if len(fotogramas) == 0:
        return None

    secuencia_uniforme = []

    # ---------------------------------------------
    # MENOS DE 75 FRAMES -> INTERPOLACIÓN
    # ---------------------------------------------

    if len(fotogramas) < NUM_FOTOGRAMAS_OBJETIVO:

        indices = np.linspace(
            0,
            len(fotogramas) - 1,
            NUM_FOTOGRAMAS_OBJETIVO
        )

        for idx in indices:

            idx_inf = int(np.floor(idx))
            idx_sup = int(np.ceil(idx))

            if idx_inf == idx_sup:

                secuencia_uniforme.extend(
                    fotogramas[idx_inf]
                )

            else:

                peso = idx - idx_inf

                frame_interpolado = [
                    (1 - peso) * a + peso * b
                    for a, b in zip(
                        fotogramas[idx_inf],
                        fotogramas[idx_sup]
                    )
                ]

                secuencia_uniforme.extend(
                    frame_interpolado
                )

    # ---------------------------------------------
    # 75 O MÁS -> MUESTREO
    # ---------------------------------------------

    else:

        indices = np.linspace(
            0,
            len(fotogramas) - 1,
            NUM_FOTOGRAMAS_OBJETIVO,
            dtype=int
        )

        for idx in indices:

            secuencia_uniforme.extend(
                fotogramas[idx]
            )

    return secuencia_uniforme


# -------------------------------------------------
# 5. BUCLE PRINCIPAL
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

    contador_frames += 1


    # -------------------------------------------------
    # 6. DETECTAR MANO
    # -------------------------------------------------

    if resultado.hand_landmarks:

        mano = resultado.hand_landmarks[0]

        tipo_mano = (
            resultado.handedness[0][0].category_name
        )

        muneca = mano[0]

        x0 = muneca.x
        y0 = muneca.y
        z0 = muneca.z


        # -------------------------------------------------
        # GUARDAR LOS LANDMARKS CRUDOS EN EL BUFFER
        # -------------------------------------------------
        #
        # IMPORTANTE:
        # Todavía NO calculamos dx/dy/dz.
        #
        # Los calcularemos tomando como referencia
        # el primer frame de la ventana.
        #
        # Así imitamos el entrenamiento.
        # -------------------------------------------------

        landmarks_actuales = []

        for punto in mano:

            landmarks_actuales.append(
                (punto.x, punto.y, punto.z)
            )

        buffer_frames.append({
            "landmarks": landmarks_actuales,
            "tipo_mano": tipo_mano
        })


        # -------------------------------------------------
        # 7. CALCULAR MOVIMIENTO DE LA VENTANA
        # -------------------------------------------------

        if len(buffer_frames) >= 2:

            primer_frame = buffer_frames[0]
            ultimo_frame = buffer_frames[-1]

            muneca_inicio = primer_frame["landmarks"][0]
            muneca_fin = ultimo_frame["landmarks"][0]

            xi, yi, zi = muneca_inicio
            xf, yf, zf = muneca_fin

            dx_mov = xf - xi
            dy_mov = yf - yi
            dz_mov = zf - zi

            # Para el umbral solamente nos interesa
            # cuánto se desplazó la muñeca.
            movimiento_actual = math.sqrt(
                dx_mov**2 +
                dy_mov**2 +
                dz_mov**2
            )

        else:

            movimiento_actual = 0.0


        # -------------------------------------------------
        # 8. ¿HAY SUFICIENTE MOVIMIENTO?
        # -------------------------------------------------

        hay_movimiento = (
            len(buffer_frames) >= MIN_FRAMES
            and movimiento_actual >= UMBRAL_MOVIMIENTO
        )


        # -------------------------------------------------
        # 9. HACER PREDICCIÓN
        # -------------------------------------------------

        if (
            hay_movimiento
            and contador_frames % INTERVALO_PREDICCION == 0
        ):

            # ---------------------------------------------
            # La muñeca del PRIMER FRAME es nuestro origen.
            # ---------------------------------------------

            primer_frame = buffer_frames[0]

            xi, yi, zi = (
                primer_frame["landmarks"][0]
            )

            fotogramas_secuencia = []


            # ---------------------------------------------
            # PROCESAR TODOS LOS FRAMES DE LA VENTANA
            # ---------------------------------------------

            for datos_frame in buffer_frames:

                landmarks = datos_frame["landmarks"]
                tipo = datos_frame["tipo_mano"]

                x0, y0, z0 = landmarks[0]


                # -----------------------------------------
                # MOVIMIENTO DE MUÑECA
                # -----------------------------------------

                dx = x0 - xi
                dy = y0 - yi
                dz = z0 - zi

                if tipo == "Left":
                    dx = -dx


                # -----------------------------------------
                # LANDMARKS RELATIVOS A MUÑECA ACTUAL
                # -----------------------------------------

                landmarks_relativos = []

                for x, y, z in landmarks:

                    x_rel = x - x0

                    if tipo == "Left":
                        x_rel = -x_rel

                    y_rel = y - y0
                    z_rel = z - z0

                    landmarks_relativos.append(
                        [x_rel, y_rel, z_rel]
                    )


                # -----------------------------------------
                # NORMALIZACIÓN DE ESCALA
                # -----------------------------------------

                dist_max = max(
                    math.sqrt(
                        x**2 +
                        y**2 +
                        z**2
                    )
                    for x, y, z
                    in landmarks_relativos
                )


                if dist_max > 0:

                    vector_fotograma = []

                    # 63 características de forma
                    for x, y, z in landmarks_relativos:

                        vector_fotograma.extend([
                            x / dist_max,
                            y / dist_max,
                            z / dist_max
                        ])

                    # 3 características de movimiento
                    vector_fotograma.extend([
                        dx,
                        dy,
                        dz
                    ])

                    # 66 características
                    fotogramas_secuencia.append(
                        vector_fotograma
                    )


            # -------------------------------------------------
            # 10. CONVERTIR LA VENTANA A 75 FRAMES
            # -------------------------------------------------

            secuencia_uniforme = convertir_a_75_frames(
                fotogramas_secuencia
            )


            # -------------------------------------------------
            # 11. PREDICCIÓN
            # -------------------------------------------------

            if (
                secuencia_uniforme is not None
                and len(secuencia_uniforme) == 4950
            ):

                prediccion = modelo.predict(
                    [secuencia_uniforme]
                )[0]

                historial_predicciones.append(
                    prediccion
                )

                print(
                    "Movimiento:",
                    round(movimiento_actual, 4),
                    "| Predicción candidata:",
                    prediccion
                )


                # -------------------------------------------------
                # 12. ESTABILIZACIÓN
                # -------------------------------------------------

                conteo = Counter(
                    historial_predicciones
                )

                letra_mas_comun, repeticiones = (
                    conteo.most_common(1)[0]
                )

                if (
                    repeticiones
                    >= NUM_PREDICCIONES_ESTABLES
                ):

                    letra_predicha = letra_mas_comun
                    color_texto = (0, 255, 0)

                else:

                    letra_predicha = "Analizando..."
                    color_texto = (0, 255, 255)


        # -------------------------------------------------
        # SIN MOVIMIENTO
        # -------------------------------------------------

        elif not hay_movimiento:

            historial_predicciones.clear()

            letra_predicha = "Esperando movimiento..."
            color_texto = (0, 0, 255)


        # -------------------------------------------------
        # DIBUJAR LANDMARKS
        # -------------------------------------------------

        for punto in mano:

            x_pixel = int(
                punto.x * ancho
            )

            y_pixel = int(
                punto.y * alto
            )

            cv2.circle(
                frame,
                (x_pixel, y_pixel),
                4,
                (255, 0, 0),
                -1
            )


    # -------------------------------------------------
    # 13. NO SE DETECTÓ MANO
    # -------------------------------------------------

    else:

        buffer_frames.clear()
        historial_predicciones.clear()

        movimiento_actual = 0.0

        letra_predicha = "Sin mano"
        color_texto = (0, 0, 255)


    # -------------------------------------------------
    # 14. INTERFAZ
    # -------------------------------------------------

    cv2.rectangle(
        frame,
        (20, 20),
        (550, 135),
        (50, 50, 50),
        -1
    )

    cv2.putText(
        frame,
        f"Letra: {letra_predicha}",
        (35, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color_texto,
        2
    )

    cv2.putText(
        frame,
        f"Movimiento: {movimiento_actual:.4f}",
        (35, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Buffer: {len(buffer_frames)}/{TAM_BUFFER}",
        (35, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Reconocimiento automatico | ESC = salir",
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
    # 15. TECLADO
    # -------------------------------------------------

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == 27:
        break


# -------------------------------------------------
# 16. CERRAR
# -------------------------------------------------

cap.release()
cv2.destroyAllWindows()
detector.close()

print("Programa finalizado.")