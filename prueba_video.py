import cv2
import mediapipe as mp
import pandas as pd

# -------------------------------------------------
# 1. RUTAS
# -------------------------------------------------

ruta_video = (
    "dataset/MSL-dynamic-signs-frontal-view/"
    "MSL-dynamic-signs/train/S1-J-frontal-1.mp4"
)

ruta_modelo = "hand_landmarker.task"


# -------------------------------------------------
# 2. CONFIGURAR MEDIAPIPE
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

detector = HandLandmarker.create_from_options(opciones)


# -------------------------------------------------
# 3. ABRIR VIDEO
# -------------------------------------------------

video = cv2.VideoCapture(ruta_video)

if not video.isOpened():
    print("ERROR: No se pudo abrir el video")
    exit()

fps = video.get(cv2.CAP_PROP_FPS)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

print("FPS:", fps)
print("Frames del video:", total_frames)


# -------------------------------------------------
# 4. CONEXIONES ENTRE LANDMARKS
# -------------------------------------------------

# Nos permiten dibujar el "esqueleto" de la mano
conexiones = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS


# -------------------------------------------------
# 5. DATOS
# -------------------------------------------------

datos = []
frame_num = 0


# -------------------------------------------------
# 6. PROCESAR VIDEO
# -------------------------------------------------

while True:

    ret, frame = video.read()

    if not ret:
        break

    # Convertir BGR -> RGB
    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )


    # Tiempo correspondiente al frame
    timestamp_ms = int(
        (frame_num / fps) * 1000
    )


    # Detectar landmarks
    resultado = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # -------------------------------------------------
    # 7. SI SE DETECTÓ UNA MANO
    # -------------------------------------------------
    fallo_deteccion = False
    if resultado.hand_landmarks:

        print(f"Frame {frame_num}: mano detectada")

        mano = resultado.hand_landmarks[0]

        fila = {
            "letra": "J",
            "video": "S1-J-frontal-1",
            "frame": frame_num,
            "tiempo_ms": timestamp_ms
        }


        # Guardar landmarks en el CSV
        for i, landmark in enumerate(mano):

            fila[f"x{i}"] = landmark.x
            fila[f"y{i}"] = landmark.y
            fila[f"z{i}"] = landmark.z


        datos.append(fila)


        # -------------------------------------------------
        # 8. DIBUJAR LOS 21 LANDMARKS
        # -------------------------------------------------

        alto, ancho, _ = frame.shape

        # Dibujar puntos
        for landmark in mano:

            x = int(landmark.x * ancho)
            y = int(landmark.y * alto)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )


        # Dibujar conexiones
        for conexion in conexiones:

            inicio = conexion.start
            fin = conexion.end

            landmark_inicio = mano[inicio]
            landmark_fin = mano[fin]

            x1 = int(landmark_inicio.x * ancho)
            y1 = int(landmark_inicio.y * alto)

            x2 = int(landmark_fin.x * ancho)
            y2 = int(landmark_fin.y * alto)

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )


    # -------------------------------------------------
    # 9. SI NO DETECTÓ MANO
    # -------------------------------------------------

    else:
        fallo_deteccion = True

        print(f"Frame {frame_num}: NO se detectó mano")

        cv2.putText(
            frame,
            "NO SE DETECTO MANO",
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # -------------------------------------------------
    # 10. MOSTRAR NÚMERO DE FRAME
    # -------------------------------------------------

    cv2.putText(
        frame,
        f"Frame: {frame_num}",
        (30, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )


    # -------------------------------------------------
    # 11. MOSTRAR VENTANA
    # -------------------------------------------------

    cv2.imshow(
        "Procesamiento MediaPipe",
        frame
    )

    if fallo_deteccion:
        print("\nDetenido en el frame:", frame_num)
        print("Presiona cualquier tecla para continuar...")
        cv2.waitKey(0)
    else:
        cv2.waitKey(int(1000 / fps))


    # Esperar aproximadamente según FPS
    tecla = cv2.waitKey(int(1000 / fps)) & 0xFF

    # Presionar q para salir
    if tecla == ord("q"):
        break


    frame_num += 1


# -------------------------------------------------
# 12. CERRAR TODO
# -------------------------------------------------

video.release()
detector.close()

cv2.destroyAllWindows()


# -------------------------------------------------
# 13. CREAR CSV
# -------------------------------------------------

df = pd.DataFrame(datos)

df.to_csv(
    "S1-J-frontal-1_landmarks.csv",
    index=False
)


print("\nProceso terminado")
print("Frames del video:", total_frames)
print("Frames con mano detectada:", len(df))

print("\nPrimeras filas del CSV:")
print(df.head())