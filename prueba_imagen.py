import cv2
import mediapipe as mp

ruta_imagen = "MSL-ABC/lsm-abc-B/test/V/S18-V-1-3.jpg"
ruta_modelo = "hand_landmarker.task"

imagen = cv2.imread(ruta_imagen)

if imagen is None:
    print("Error: no se pudo abrir la imagen")
    exit()

print("Imagen cargada correctamente")

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

imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

imagen_mp = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=imagen_rgb
)

resultado = detector.detect(imagen_mp)

if resultado.hand_landmarks:

    print("¡Mano detectada!")

    mano = resultado.hand_landmarks[0]
    alto, ancho, _ = imagen.shape

    for numero, punto in enumerate(mano):

        print("Landmark", numero)
        print("x:", punto.x)
        print("y:", punto.y)
        print("z:", punto.z)
        print("----------------")

        x_pixel = int(punto.x * ancho)
        y_pixel = int(punto.y * alto)

        cv2.circle(
            imagen,
            (x_pixel, y_pixel),
            5,
            (0, 255, 0),
            -1
        )
    
    print("Cantidad de landmarks:", len(mano))
    
    nombre_ventana = "Landmarks"
    cv2.imshow(nombre_ventana, imagen)
    
    while True:
        tecla = cv2.waitKey(100)
        if tecla != -1:  
            break
            
        if cv2.getWindowProperty(nombre_ventana, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
else:
    print("No se detectó ninguna mano")


detector.close()