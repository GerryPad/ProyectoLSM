import cv2
import mediapipe as mp
import joblib
import math
import numpy as np
import pandas as pd
from collections import deque


# -------------------------------------------------
# 1. CONFIGURACIÓN
# -------------------------------------------------

modelo = joblib.load("modelo_lsm_dinamico_movimiento.pkl")
ruta_modelo = "hand_landmarker.task"

NUM_FOTOGRAMAS_OBJETIVO = 75


# -------------------------------------------------
# PARÁMETROS PARA DETECTAR INICIO Y FIN
# -------------------------------------------------

# Comparamos la muñeca actual con la de hace
# algunos frames para saber si REALMENTE se mueve.
FRAMES_COMPARACION = 5

# Umbral provisional para considerar movimiento.
# Después lo ajustaremos con tus pruebas.
UMBRAL_INICIO = 0.025

# Para detectar que terminó la seña usamos un
# umbral menor.
UMBRAL_FIN = 0.012

# No basta un único frame con movimiento.
# Pedimos varios consecutivos para iniciar.
FRAMES_PARA_INICIAR = 3

# Pedimos varios frames quietos para terminar.
FRAMES_PARA_TERMINAR = 8

# Evita procesar secuencias demasiado pequeñas.
MIN_FRAMES_SENA = 12

# Evita quedarse capturando indefinidamente.
MAX_FRAMES_SENA = 120


# -------------------------------------------------
# 2. MEDIAPIPE
# -------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


opciones = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=ruta_modelo
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

detector = HandLandmarker.create_from_options(
    opciones
)


# -------------------------------------------------
# 3. CÁMARA
# -------------------------------------------------

cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


if not cap.isOpened():

    print("No se pudo abrir la cámara.")
    exit()


print(
    "Reconocimiento dinámico automático iniciado."
)

print(
    "Mueve la mano para realizar una seña."
)

print(
    "Presiona ESC para salir."
)


# -------------------------------------------------
# 4. VARIABLES DE ESTADO
# -------------------------------------------------

timestamp = 0


# Estados posibles:
#
# ESPERANDO
# CAPTURANDO

estado = "ESPERANDO"


# -------------------------------------------------
# Historial corto de muñecas.
#
# Solo sirve para determinar si actualmente
# hay movimiento.
# -------------------------------------------------

historial_munecas = deque(
    maxlen=FRAMES_COMPARACION + 1
)


# -------------------------------------------------
# Frames que pertenecen a la seña actual
# -------------------------------------------------

secuencia_actual = []


# -------------------------------------------------
# Contadores
# -------------------------------------------------

frames_movimiento = 0
frames_quietos = 0


# -------------------------------------------------
# Información mostrada
# -------------------------------------------------

movimiento_actual = 0.0

letra_predicha = "Esperando..."

color_texto = (
    0,
    0,
    255
)


# -------------------------------------------------
# 5. FUNCIÓN PARA CONVERTIR A 75 FRAMES
# -------------------------------------------------

def convertir_a_75_frames(fotogramas):

    if len(fotogramas) == 0:
        return None


    secuencia_uniforme = []


    # ---------------------------------------------
    # MENOS DE 75 -> INTERPOLAR
    # ---------------------------------------------

    if (
        len(fotogramas)
        < NUM_FOTOGRAMAS_OBJETIVO
    ):

        indices = np.linspace(
            0,
            len(fotogramas) - 1,
            NUM_FOTOGRAMAS_OBJETIVO
        )


        for idx in indices:

            idx_inf = int(
                np.floor(idx)
            )

            idx_sup = int(
                np.ceil(idx)
            )


            if idx_inf == idx_sup:

                secuencia_uniforme.extend(
                    fotogramas[idx_inf]
                )

            else:

                peso = idx - idx_inf

                frame_interpolado = [

                    (1 - peso) * a
                    + peso * b

                    for a, b in zip(
                        fotogramas[idx_inf],
                        fotogramas[idx_sup]
                    )

                ]

                secuencia_uniforme.extend(
                    frame_interpolado
                )


    # ---------------------------------------------
    # 75 O MÁS -> MUESTREAR
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
# 6. FUNCIÓN PARA PROCESAR UNA SEÑA COMPLETA
# -------------------------------------------------

def procesar_sena(secuencia):

    if len(secuencia) < MIN_FRAMES_SENA:

        print(
            "Secuencia demasiado corta. Ignorada."
        )

        return None


    print(
        "\nProcesando seña de",
        len(secuencia),
        "frames..."
    )


    # -------------------------------------------------
    # MUÑECA INICIAL
    # -------------------------------------------------
    #
    # Igual que durante el entrenamiento:
    # el primer frame será nuestro origen.
    # -------------------------------------------------

    primer_frame = secuencia[0]

    xi, yi, zi = (
        primer_frame["landmarks"][0]
    )


    fotogramas_procesados = []


    # -------------------------------------------------
    # PROCESAR CADA FRAME
    # -------------------------------------------------

    for datos_frame in secuencia:

        landmarks = (
            datos_frame["landmarks"]
        )

        tipo_mano = (
            datos_frame["tipo_mano"]
        )


        # Muñeca actual
        x0, y0, z0 = landmarks[0]


        # -------------------------------------------------
        # TRAYECTORIA DE LA MUÑECA
        # -------------------------------------------------

        dx = x0 - xi
        dy = y0 - yi
        dz = z0 - zi


        if tipo_mano == "Left":

            dx = -dx


        # -------------------------------------------------
        # LANDMARKS RELATIVOS A LA MUÑECA ACTUAL
        # -------------------------------------------------

        landmarks_relativos = []


        for x, y, z in landmarks:

            x_rel = x - x0

            if tipo_mano == "Left":

                x_rel = -x_rel


            y_rel = y - y0
            z_rel = z - z0


            landmarks_relativos.append(
                [
                    x_rel,
                    y_rel,
                    z_rel
                ]
            )


        # -------------------------------------------------
        # NORMALIZACIÓN
        # -------------------------------------------------

        dist_max = max(

            math.sqrt(
                x**2
                + y**2
                + z**2
            )

            for x, y, z
            in landmarks_relativos
        )


        if dist_max > 0:

            vector_fotograma = []


            # ---------------------------------------------
            # 63 CARACTERÍSTICAS DE FORMA
            # ---------------------------------------------

            for x, y, z in landmarks_relativos:

                vector_fotograma.extend([
                    x / dist_max,
                    y / dist_max,
                    z / dist_max
                ])


            # ---------------------------------------------
            # 3 CARACTERÍSTICAS DE MOVIMIENTO
            # ---------------------------------------------

            vector_fotograma.extend([
                dx,
                dy,
                dz
            ])


            # Total = 66
            fotogramas_procesados.append(
                vector_fotograma
            )


    # -------------------------------------------------
    # CONVERTIR A 75 FRAMES
    # -------------------------------------------------

    secuencia_uniforme = (
        convertir_a_75_frames(
            fotogramas_procesados
        )
    )


    if secuencia_uniforme is None:

        return None


    print(
        "Características:",
        len(secuencia_uniforme)
    )


    # -------------------------------------------------
    # VERIFICACIÓN
    # -------------------------------------------------

    if len(secuencia_uniforme) != 4950:

        print(
            "Error: se esperaban 4950 características."
        )

        return None


    # -------------------------------------------------
    # PREDICCIÓN
    # -------------------------------------------------
    #
    # Creamos DataFrame usando los mismos nombres
    # del entrenamiento para evitar el warning:
    #
    # "X does not have valid feature names"
    # -------------------------------------------------

    if hasattr(
        modelo,
        "feature_names_in_"
    ):

        entrada = pd.DataFrame(
            [secuencia_uniforme],
            columns=modelo.feature_names_in_
        )

    else:

        entrada = [
            secuencia_uniforme
        ]


    prediccion = modelo.predict(
        entrada
    )[0]


    # -------------------------------------------------
    # CONFIANZA
    # -------------------------------------------------

    confianza = None


    if hasattr(
        modelo,
        "predict_proba"
    ):

        probabilidades = (
            modelo.predict_proba(
                entrada
            )[0]
        )

        confianza = float(
            np.max(probabilidades)
        )


    return (
        prediccion,
        confianza
    )


# -------------------------------------------------
# 7. BUCLE PRINCIPAL
# -------------------------------------------------

while cap.isOpened():

    ret, frame = cap.read()


    if not ret:

        break


    # -------------------------------------------------
    # ESPEJO
    # -------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    alto, ancho, _ = (
        frame.shape
    )


    # -------------------------------------------------
    # CONVERTIR PARA MEDIAPIPE
    # -------------------------------------------------

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    imagen_mp = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )


    timestamp += int(
        1000 / 30
    )


    resultado = (
        detector.detect_for_video(
            imagen_mp,
            timestamp
        )
    )


    # -------------------------------------------------
    # 8. MANO DETECTADA
    # -------------------------------------------------

    if resultado.hand_landmarks:

        mano = (
            resultado.hand_landmarks[0]
        )


        tipo_mano = (
            resultado
            .handedness[0][0]
            .category_name
        )


        # -------------------------------------------------
        # MUÑECA ACTUAL
        # -------------------------------------------------

        muneca = mano[0]

        x0 = muneca.x
        y0 = muneca.y
        z0 = muneca.z


        # -------------------------------------------------
        # GUARDAR MUÑECA EN HISTORIAL CORTO
        # -------------------------------------------------

        historial_munecas.append(
            (
                x0,
                y0,
                z0
            )
        )


        # -------------------------------------------------
        # CALCULAR MOVIMIENTO RECIENTE
        # -------------------------------------------------

        if (
            len(historial_munecas)
            == FRAMES_COMPARACION + 1
        ):

            xa, ya, za = (
                historial_munecas[0]
            )

            xb, yb, zb = (
                historial_munecas[-1]
            )


            movimiento_actual = (
                math.sqrt(
                    (xb - xa)**2
                    + (yb - ya)**2
                    + (zb - za)**2
                )
            )

        else:

            movimiento_actual = 0.0


        # -------------------------------------------------
        # PREPARAR DATOS CRUDOS DEL FRAME
        # -------------------------------------------------

        landmarks_actuales = []


        for punto in mano:

            landmarks_actuales.append(
                (
                    punto.x,
                    punto.y,
                    punto.z
                )
            )


        datos_actuales = {

            "landmarks":
                landmarks_actuales,

            "tipo_mano":
                tipo_mano

        }


        # =================================================
        # 9. ESTADO: ESPERANDO
        # =================================================

        if estado == "ESPERANDO":

            # ---------------------------------------------
            # ¿HAY MOVIMIENTO REAL?
            # ---------------------------------------------

            if (
                movimiento_actual
                >= UMBRAL_INICIO
            ):

                frames_movimiento += 1

            else:

                frames_movimiento = 0


            # ---------------------------------------------
            # VARIOS FRAMES CON MOVIMIENTO
            # -> INICIA SEÑA
            # ---------------------------------------------

            if (
                frames_movimiento
                >= FRAMES_PARA_INICIAR
            ):

                print(
                    "\n>>> INICIO DE SEÑA"
                )


                estado = "CAPTURANDO"

                secuencia_actual = []

                frames_quietos = 0

                frames_movimiento = 0


                # -----------------------------------------
                # IMPORTANTE
                #
                # Guardamos también algunos frames
                # anteriores al instante exacto en que
                # detectamos el inicio.
                # -----------------------------------------

                secuencia_actual.append(
                    datos_actuales
                )


                letra_predicha = (
                    "Capturando..."
                )

                color_texto = (
                    0,
                    255,
                    255
                )


        # =================================================
        # 10. ESTADO: CAPTURANDO
        # =================================================

        elif estado == "CAPTURANDO":

            # ---------------------------------------------
            # GUARDAMOS TODOS LOS FRAMES
            # ---------------------------------------------

            secuencia_actual.append(
                datos_actuales
            )


            # ---------------------------------------------
            # ¿LA MANO ESTÁ QUIETA?
            # ---------------------------------------------

            if (
                movimiento_actual
                <= UMBRAL_FIN
            ):

                frames_quietos += 1

            else:

                frames_quietos = 0


            # ---------------------------------------------
            # TERMINAR POR QUIETUD
            # ---------------------------------------------

            terminar_por_quietud = (

                frames_quietos
                >= FRAMES_PARA_TERMINAR

                and

                len(secuencia_actual)
                >= MIN_FRAMES_SENA

            )


            # ---------------------------------------------
            # TERMINAR POR SEGURIDAD
            # ---------------------------------------------

            terminar_por_maximo = (

                len(secuencia_actual)
                >= MAX_FRAMES_SENA

            )


            # ---------------------------------------------
            # FIN DE SEÑA
            # ---------------------------------------------

            if (
                terminar_por_quietud
                or terminar_por_maximo
            ):

                print(
                    "<<< FIN DE SEÑA"
                )


                # -----------------------------------------
                # Eliminamos algunos frames quietos
                # del final.
                # -----------------------------------------

                if terminar_por_quietud:

                    frames_a_quitar = (
                        FRAMES_PARA_TERMINAR
                        - 1
                    )


                    if (
                        len(secuencia_actual)
                        > frames_a_quitar
                    ):

                        secuencia_para_procesar = (
                            secuencia_actual[
                                :-frames_a_quitar
                            ]
                        )

                    else:

                        secuencia_para_procesar = (
                            secuencia_actual.copy()
                        )

                else:

                    secuencia_para_procesar = (
                        secuencia_actual.copy()
                    )


                # -----------------------------------------
                # CLASIFICAR
                # -----------------------------------------

                resultado_prediccion = (
                    procesar_sena(
                        secuencia_para_procesar
                    )
                )


                if (
                    resultado_prediccion
                    is not None
                ):

                    letra, confianza = (
                        resultado_prediccion
                    )


                    letra_predicha = letra

                    color_texto = (
                        0,
                        255,
                        0
                    )


                    if confianza is not None:

                        print(
                            "Letra detectada:",
                            letra,
                            "| Confianza:",
                            round(
                                confianza * 100,
                                2
                            ),
                            "%"
                        )

                    else:

                        print(
                            "Letra detectada:",
                            letra
                        )


                else:

                    letra_predicha = (
                        "No reconocida"
                    )

                    color_texto = (
                        0,
                        0,
                        255
                    )


                # -----------------------------------------
                # REINICIAR PARA LA SIGUIENTE SEÑA
                # -----------------------------------------

                estado = "ESPERANDO"

                secuencia_actual = []

                historial_munecas.clear()

                frames_quietos = 0

                frames_movimiento = 0


        # -------------------------------------------------
        # 11. DIBUJAR LANDMARKS
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
                (
                    x_pixel,
                    y_pixel
                ),
                4,
                (255, 0, 0),
                -1
            )


    # -------------------------------------------------
    # 12. NO HAY MANO
    # -------------------------------------------------

    else:

        historial_munecas.clear()

        movimiento_actual = 0.0


        # Si todavía no estábamos capturando,
        # simplemente seguimos esperando.

        if estado == "ESPERANDO":

            frames_movimiento = 0


    # -------------------------------------------------
    # 13. INTERFAZ
    # -------------------------------------------------

    cv2.rectangle(
        frame,
        (20, 20),
        (600, 175),
        (50, 50, 50),
        -1
    )


    cv2.putText(
        frame,
        f"Letra: {letra_predicha}",
        (35, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color_texto,
        2
    )


    cv2.putText(
        frame,
        f"Estado: {estado}",
        (35, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Movimiento reciente: {movimiento_actual:.4f}",
        (35, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Frames sena: {len(secuencia_actual)}",
        (35, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "Automatico | ESC = salir",
        (20, alto - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "Reconocimiento LSM Dinamico Automatico",
        frame
    )


    # -------------------------------------------------
    # 14. TECLADO
    # -------------------------------------------------

    tecla = (
        cv2.waitKey(1)
        & 0xFF
    )


    if tecla == 27:

        break


# -------------------------------------------------
# 15. CERRAR
# -------------------------------------------------

cap.release()

cv2.destroyAllWindows()

detector.close()

print(
    "Programa finalizado."
)