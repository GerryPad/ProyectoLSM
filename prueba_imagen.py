# TAREA. Hacer el repositorio de GitHub
import cv2
import mediapipe as mp


# -------------------------------------------------
# 1. RUTAS
# -------------------------------------------------

ruta_imagen = "MSL-ABC/lsm-abc-B/test/V/S18-V-1-3.jpg"
ruta_modelo = "hand_landmarker.task"


# -------------------------------------------------
# 2. ABRIR LA IMAGEN CON OPENCV
# -------------------------------------------------

imagen = cv2.imread(ruta_imagen)

if imagen is None:
    print("Error: no se pudo abrir la imagen")
    exit()

print("Imagen cargada correctamente")


# -------------------------------------------------
# 3. PREPARAR MEDIAPIPE
# -------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

opciones = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=ruta_modelo
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=1
)

detector = HandLandmarker.create_from_options(opciones)


# -------------------------------------------------
# 4. CONVERTIR BGR → RGB
# -------------------------------------------------

imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)


# -------------------------------------------------
# 5. CONVERTIRLA A UNA IMAGEN DE MEDIAPIPE
# -------------------------------------------------

imagen_mp = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=imagen_rgb
)


# -------------------------------------------------
# 6. DETECTAR LA MANO
# -------------------------------------------------

resultado = detector.detect(imagen_mp)


# -------------------------------------------------
# 7. COMPROBAR SI ENCONTRÓ UNA MANO
# -------------------------------------------------

if resultado.hand_landmarks:

    print("¡Mano detectada!")

    mano = resultado.hand_landmarks[0]

    for numero, punto in enumerate(mano):

        print("Landmark", numero)

        print("x:", punto.x)
        print("y:", punto.y)
        print("z:", punto.z)

        print("----------------")
        alto, ancho, _ = imagen.shape

        x_pixel = int(punto.x * ancho)
        y_pixel = int(punto.y * alto)

        cv2.circle(
            imagen,
            (x_pixel, y_pixel),
            5,
            (0, 255, 0),
            -1
        )
    cv2.imshow("Landmarks", imagen)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Cantidad de landmarks:", len(mano))

else:

    print("No se detectó ninguna mano")


detector.close()