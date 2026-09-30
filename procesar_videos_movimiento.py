import os
import cv2
import mediapipe as mp
import pandas as pd
import math
import numpy as np

# -------------------------------------------------
# 1. CONFIGURACIÓN DE RUTAS Y MEDIAPIPE
# -------------------------------------------------

carpeta_videos = (
    "dataset/MSL-dynamic-signs-profile/"
    "MSL dynamic-profile-signs/Z"
)

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

NUM_FOTOGRAMAS_OBJETIVO = 75

datos_videos = []
etiquetas = []

print(f"Procesando videos de la carpeta: {carpeta_videos}")

# -------------------------------------------------
# 2. CONTADORES
# -------------------------------------------------

contador_videos = 0
videos_procesados = 0
videos_estirados = 0


# -------------------------------------------------
# 3. PROCESAMIENTO DE VIDEOS
# -------------------------------------------------

if os.path.exists(carpeta_videos):

    for archivo in os.listdir(carpeta_videos):

        if archivo.lower().endswith(
            ('.mp4', '.avi', '.mov', '.mkv')
        ):

            ruta_video = os.path.join(
                carpeta_videos,
                archivo
            )

            # Ejemplo:
            # S1-J-frontal-1.mp4 -> J

            partes = archivo.split('-')

            if len(partes) >= 2:
                letra = partes[1].upper()
            else:
                continue

            cap = cv2.VideoCapture(ruta_video)

            fotogramas_video = []

            # -----------------------------------------
            # Posición inicial de la muñeca
            # -----------------------------------------

            muneca_inicial = None

            detector = HandLandmarker.create_from_options(
                opciones
            )

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

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

                    mano = resultado.hand_landmarks[0]

                    tipo_mano = (
                        resultado.handedness[0][0].category_name
                    )

                    # ---------------------------------
                    # MUÑECA ACTUAL
                    # ---------------------------------

                    muneca = mano[0]

                    x0 = muneca.x
                    y0 = muneca.y
                    z0 = muneca.z

                    # La primera muñeca detectada
                    # será nuestro punto de referencia

                    if muneca_inicial is None:

                        muneca_inicial = (
                            x0,
                            y0,
                            z0
                        )

                    xi, yi, zi = muneca_inicial

                    # ---------------------------------
                    # MOVIMIENTO DE LA MUÑECA
                    # ---------------------------------

                    dx = x0 - xi
                    dy = y0 - yi
                    dz = z0 - zi

                    # Si la mano es izquierda,
                    # aplicamos el mismo espejo en X

                    if tipo_mano == "Left":
                        dx = -dx

                    # ---------------------------------
                    # LANDMARKS RELATIVOS A LA MUÑECA
                    # ---------------------------------

                    landmarks_relativos = []

                    for punto in mano:

                        x_rel = punto.x - x0

                        if tipo_mano == "Left":
                            x_rel = -x_rel

                        y_rel = punto.y - y0
                        z_rel = punto.z - z0

                        landmarks_relativos.append(
                            [x_rel, y_rel, z_rel]
                        )

                    # ---------------------------------
                    # NORMALIZACIÓN DE LA FORMA
                    # ---------------------------------

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

                        # 63 características:
                        # forma normalizada de la mano

                        for x, y, z in landmarks_relativos:

                            vector_fotograma.extend([
                                x / dist_max,
                                y / dist_max,
                                z / dist_max
                            ])

                        # ---------------------------------
                        # 3 CARACTERÍSTICAS NUEVAS
                        # trayectoria de la muñeca
                        # ---------------------------------

                        vector_fotograma.extend([
                            dx,
                            dy,
                            dz
                        ])

                        # Total:
                        # 63 + 3 = 66 por frame

                        fotogramas_video.append(
                            vector_fotograma
                        )

            cap.release()
            detector.close()

            # -----------------------------------------
            # 4. CONVERTIR A 75 FRAMES
            # -----------------------------------------

            if len(fotogramas_video) > 0:

                secuencia_uniforme = []

                if (
                    len(fotogramas_video)
                    < NUM_FOTOGRAMAS_OBJETIVO
                ):

                    videos_estirados += 1

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

                # -------------------------------------
                # VERIFICACIÓN
                # -------------------------------------

                if len(secuencia_uniforme) != 4950:

                    print(
                        f"ADVERTENCIA: {archivo} generó "
                        f"{len(secuencia_uniforme)} características"
                    )

                    continue

                datos_videos.append(
                    secuencia_uniforme
                )

                etiquetas.append(
                    letra
                )

                videos_procesados += 1

            contador_videos += 1

            if contador_videos % 10 == 0:

                print(
                    f"Procesados "
                    f"{contador_videos} videos..."
                )

else:

    print(
        "La ruta especificada no existe."
    )


# -------------------------------------------------
# 5. CREAR NOMBRES DE COLUMNAS
# -------------------------------------------------

if len(datos_videos) > 0:

    columnas = []

    for f in range(NUM_FOTOGRAMAS_OBJETIVO):

        # 21 landmarks × XYZ

        for i in range(21):

            columnas.extend([
                f'f{f}_x{i}',
                f'f{f}_y{i}',
                f'f{f}_z{i}'
            ])

        # Movimiento de muñeca de este frame

        columnas.extend([
            f'f{f}_muneca_dx',
            f'f{f}_muneca_dy',
            f'f{f}_muneca_dz'
        ])

    # -------------------------------------------------
    # 6. CREAR DATAFRAME
    # -------------------------------------------------

    df_dinamico = pd.DataFrame(
        datos_videos,
        columns=columnas
    )

    df_dinamico['etiqueta'] = etiquetas

    # -------------------------------------------------
    # 7. GUARDAR CSV NUEVO
    # -------------------------------------------------

    archivo_salida = (
        "dataset_landmarks_dinamicos_Z.csv"
    )

    df_dinamico.to_csv(
        archivo_salida,
        index=False
    )

    print("\n¡Proceso dinámico finalizado!")

    print(
        "Total de videos:",
        videos_procesados
    )

    print(
        "Videos interpolados:",
        videos_estirados
    )

    print(
        "Características por video:",
        len(columnas)
    )

    print(
        "Archivo generado:",
        archivo_salida
    )

else:

    print(
        "No se pudieron procesar videos válidos."
    )