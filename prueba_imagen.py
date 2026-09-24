import cv2
import mediapipe as mp

CONEXIONES = [
    # Pulgar
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Índice
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Medio
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Anular
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Meñique
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Cerrar la palma
    (0, 17)
]

# Ruta
ruta_imagen = "dataset/MSL-ABC/lsm-abc-A/train/A/S1-A-4-0.jpg"
ruta_modelo = "hand_landmarker.task"

# abrimos la imagen con OPENCV
imagen = cv2.imread(ruta_imagen)

if imagen is None:
    print("Error: no se pudo abrir la imagen")
    exit()

print("Imagen cargada correctamente")


# preparamos mediaPipe
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

# 4. CONVERTIR BGR → RGB
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

# 5. CONVERTIRLA A UNA IMAGEN DE MEDIAPIPE
imagen_mp = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=imagen_rgb
)

# detecta la mano
resultado = detector.detect(imagen_mp)


# encontro mano?
if resultado.hand_landmarks:

    print("¡Mano detectada!")

    # Primera mano detectada
    mano = resultado.hand_landmarks[0]

    # Dimensiones de la imagen
    alto, ancho, _ = imagen.shape
    muneca = mano[0]#sera nuesto puto de referencia
    x0 = muneca.x
    y0 = muneca.y
    z0 = muneca.z

    #resta ante la referencia(normalizar)
    for numero, punto in enumerate(mano):

        x_relativo = punto.x - x0
        y_relativo = punto.y - y0
        z_relativo = punto.z - z0

        print("Landmark", numero)

        print("Original:")
        print(punto.x, punto.y, punto.z)

        print("Relativo:")
        print(x_relativo, y_relativo, z_relativo)

        print("----------------")

    #muneca,p.pulgar,indice,medio,anular,menique
    puntos_importantes = [0, 4, 8, 12, 16, 20]

    for numero in puntos_importantes:

        punto = mano[numero]

        x_relativo = punto.x - x0
        y_relativo = punto.y - y0
        z_relativo = punto.z - z0

        print(
            numero,
            "x:", round(x_relativo, 3),
            "y:", round(y_relativo, 3),
            "z:", round(z_relativo, 3)
        )

    # dibuja lineas
    for inicio, fin in CONEXIONES:

        punto_inicio = mano[inicio]
        punto_fin = mano[fin]

        x1 = int(punto_inicio.x * ancho)
        y1 = int(punto_inicio.y * alto)

        x2 = int(punto_fin.x * ancho)
        y2 = int(punto_fin.y * alto)

        cv2.line(
            imagen,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )


    # dibuja landmarks
    for numero, punto in enumerate(mano):

        x = int(punto.x * ancho)
        y = int(punto.y * alto)

        cv2.circle(
            imagen,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )


    # imagen
    cv2.imshow("Esqueleto de la mano", imagen)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:

    print("No se detectó ninguna mano")

detector.close()