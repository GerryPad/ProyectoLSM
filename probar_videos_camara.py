import os
import cv2
import mediapipe as mp
import joblib
import math
import numpy as np
import pandas as pd


# -------------------------------------------------
# 1. CONFIGURACIÓN
# -------------------------------------------------

CARPETA_VIDEOS = "A_videos"
ARCHIVO_MODELO = "modelo_lsm_dinamico.pkl"
RUTA_MEDIAPIPE = "hand_landmarker.task"

NUM_FOTOGRAMAS_OBJETIVO = 75


print("Cargando modelo...")
modelo = joblib.load(ARCHIVO_MODELO)

print("Clases del modelo:")
print(modelo.classes_)


# -------------------------------------------------
# 2. CONFIGURAR MEDIAPIPE
# -------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

opciones = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=RUTA_MEDIAPIPE
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1
)


# -------------------------------------------------
# 3. FUNCIÓN PARA PROCESAR UN VIDEO
# -------------------------------------------------

def procesar_video(ruta_video):

    cap = cv2.VideoCapture(ruta_video)

    fotogramas_video = []
    frames_totales = 0
    frames_con_mano = 0

    detector = HandLandmarker.create_from_options(opciones)

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frames_totales += 1

        # IMPORTANTE:
        # No hacemos cv2.flip() porque procesar_videos.py
        # tampoco lo hizo con el dataset original.

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        imagen_mp = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        resultado = detector.detect(imagen_mp)

        if resultado.hand_landmarks:

            frames_con_mano += 1

            mano = resultado.hand_landmarks[0]

            tipo_mano = (
                resultado.handedness[0][0].category_name
            )

            # -----------------------------------------
            # Normalización respecto a la muñeca
            # -----------------------------------------

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

                landmarks_relativos.append(
                    [x_rel, y_rel, z_rel]
                )

            # -----------------------------------------
            # Normalización de escala
            # -----------------------------------------

            dist_max = max(
                math.sqrt(
                    x**2 + y**2 + z**2
                )
                for x, y, z
                in landmarks_relativos
            )

            if dist_max > 0:

                vector_fotograma = []

                for x, y, z in landmarks_relativos:

                    vector_fotograma.extend([
                        x / dist_max,
                        y / dist_max,
                        z / dist_max
                    ])

                fotogramas_video.append(
                    vector_fotograma
                )

    cap.release()
    detector.close()

    # ---------------------------------------------
    # Verificar que MediaPipe encontró la mano
    # ---------------------------------------------

    if len(fotogramas_video) == 0:

        return None, frames_totales, frames_con_mano

    # ---------------------------------------------
    # Convertir a exactamente 75 frames
    # ---------------------------------------------

    secuencia_uniforme = []

    if len(fotogramas_video) < NUM_FOTOGRAMAS_OBJETIVO:

        indices = np.linspace(
            0,
            len(fotogramas_video) - 1,
            NUM_FOTOGRAMAS_OBJETIVO
        )

        for idx in indices:

            idx_inf = int(np.floor(idx))
            idx_sup = int(np.ceil(idx))

            if idx_inf == idx_sup:

                secuencia_uniforme.extend(
                    fotogramas_video[idx_inf]
                )

            else:

                peso = idx - idx_inf

                frame_interpolado = [
                    (1 - peso) * a + peso * b
                    for a, b in zip(
                        fotogramas_video[idx_inf],
                        fotogramas_video[idx_sup]
                    )
                ]

                secuencia_uniforme.extend(
                    frame_interpolado
                )

    else:

        indices = np.linspace(
            0,
            len(fotogramas_video) - 1,
            NUM_FOTOGRAMAS_OBJETIVO,
            dtype=int
        )

        for idx in indices:

            secuencia_uniforme.extend(
                fotogramas_video[idx]
            )

    return (
        secuencia_uniforme,
        frames_totales,
        frames_con_mano
    )


# -------------------------------------------------
# 4. PROCESAR TODOS LOS VIDEOS
# -------------------------------------------------

resultados = []

archivos = sorted(os.listdir(CARPETA_VIDEOS))

for archivo in archivos:

    if not archivo.lower().endswith(
        (".mp4", ".avi", ".mov", ".mkv")
    ):
        continue

    ruta = os.path.join(
        CARPETA_VIDEOS,
        archivo
    )

    # ---------------------------------------------
    # Obtener letra real desde el nombre
    #
    # cam_K_1.mp4 -> K
    # cam_Q_2.mp4 -> Q
    # ---------------------------------------------

    partes = archivo.split("_")

    if len(partes) < 2:
        print(
            f"No pude obtener la letra de {archivo}"
        )
        continue

    letra_real = partes[1].upper()

    print("\n======================================")
    print("VIDEO:", archivo)
    print("Letra real:", letra_real)

    secuencia, total, con_mano = procesar_video(ruta)

    print("Frames totales:", total)
    print("Frames con mano:", con_mano)

    if secuencia is None:

        print("ERROR: MediaPipe no encontró la mano.")
        continue

    print(
        "Características generadas:",
        len(secuencia)
    )

    # ---------------------------------------------
    # 5. PREDICCIÓN
    # ---------------------------------------------

    X_video = np.array(
        secuencia
    ).reshape(1, -1)

    prediccion = modelo.predict(X_video)[0]

    probabilidades = modelo.predict_proba(
        X_video
    )[0]

    print("\nPREDICCIÓN:", prediccion)

    # ---------------------------------------------
    # Mostrar probabilidades ordenadas
    # ---------------------------------------------

    pares = list(
        zip(
            modelo.classes_,
            probabilidades
        )
    )

    pares.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print("\nProbabilidades:")

    for letra, probabilidad in pares:

        print(
            f"   {letra}: "
            f"{probabilidad * 100:.2f}%"
        )

    correcta = (
        letra_real == prediccion
    )

    if correcta:
        print("\n✓ CORRECTA")
    else:
        print(
            f"\n✗ ERROR: era {letra_real} "
            f"pero predijo {prediccion}"
        )

    resultados.append({
        "video": archivo,
        "real": letra_real,
        "prediccion": prediccion,
        "correcta": correcta,
        "confianza": max(probabilidades) * 100,
        "frames_totales": total,
        "frames_con_mano": con_mano
    })


# -------------------------------------------------
# 6. RESUMEN FINAL
# -------------------------------------------------

print("\n\n======================================")
print("RESUMEN DE VIDEOS DE CÁMARA")
print("======================================")

df_resultados = pd.DataFrame(resultados)

if len(df_resultados) > 0:

    print(
        df_resultados[
            [
                "video",
                "real",
                "prediccion",
                "confianza",
                "correcta"
            ]
        ].to_string(index=False)
    )

    total = len(df_resultados)

    correctas = df_resultados[
        "correcta"
    ].sum()

    print("\n--------------------------------------")

    print("Videos evaluados:", total)
    print("Predicciones correctas:", correctas)

    print(
        f"Accuracy cámara: "
        f"{correctas / total * 100:.2f}%"
    )