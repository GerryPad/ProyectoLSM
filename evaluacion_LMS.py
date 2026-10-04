import os
import cv2
import mediapipe as mp
import joblib
import math
import numpy as np
import pandas as pd


#configuracion
CARPETA_EVALUACION = "Evaluacion"
ARCHIVO_RESULTADOS = "resultados_evaluacion.csv"
ARCHIVO_RESUMEN = "resumen_evaluacion.csv"

NUM_FOTOGRAMAS_OBJETIVO = 75

EXTENSIONES_IMAGEN = (".jpg", ".jpeg", ".png")
EXTENSIONES_VIDEO = (".mp4", ".avi", ".mov", ".mkv", ".webm")


#cargamos los modelos
print("Cargando modelos...")

modelo_estatico = joblib.load("modelo_lsm_estatico.pkl")
modelo_dinamico = joblib.load("modelo_lsm_dinamico_movimiento.pkl")
ruta_modelo = "hand_landmarker.task"

# configuramos el MediaPipe
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

# Timestamp global para MediaPipe VIDEO
timestamp = 0


#INterpolamos una secuencia a 75 frames
def convertir_a_75_frames(fotogramas):

    if not fotogramas:
        return None

    secuencia_uniforme = []

    if len(fotogramas) < NUM_FOTOGRAMAS_OBJETIVO:
        indices = np.linspace( 0, len(fotogramas) - 1, NUM_FOTOGRAMAS_OBJETIVO )

        for idx in indices:

            idx_inf = int(np.floor(idx))
            idx_sup = int(np.ceil(idx))

            if idx_inf == idx_sup:

                secuencia_uniforme.extend(fotogramas[idx_inf])

            else:
                peso = idx - idx_inf

                frame_interpolado = [
                    (1 - peso) * a + peso * b
                    for a, b in zip(
                        fotogramas[idx_inf],
                        fotogramas[idx_sup]
                    )
                ]

                secuencia_uniforme.extend(frame_interpolado)

    else:
        indices = np.linspace(0,len(fotogramas) - 1, NUM_FOTOGRAMAS_OBJETIVO, dtype=int)

        for idx in indices:
            secuencia_uniforme.extend(fotogramas[idx])

    return secuencia_uniforme


# detectamos landmarks en un frame
def detectar_mano(frame, fps=30):

    global timestamp

    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB,  data=frame_rgb)

    timestamp += int(1000 / fps)

    resultado = detector.detect_for_video( imagen_mp,timestamp)

    if not resultado.hand_landmarks:
        return None, None

    mano = resultado.hand_landmarks[0]

    tipo_mano = (resultado.handedness[0][0].category_name)

    return mano, tipo_mano


# procesamos imagen
def procesar_imagen(ruta):

    imagen = cv2.imread(ruta)

    if imagen is None:
        return None, "No se pudo abrir imagen"

    mano, tipo_mano = detectar_mano(imagen)

    if mano is None:
        return None, "Mano no detectada"

    # Muñeca = landmark 0
    muneca = mano[0]

    x0 = muneca.x
    y0 = muneca.y
    z0 = muneca.z

    landmarks_relativos = []

    # Mismo procesamiento que camara.py
    for punto in mano:

        x_relativo = punto.x - x0

        # Efecto espejo matemático para mano izquierda
        if tipo_mano == "Left":
            x_relativo = -x_relativo

        y_relativo = punto.y - y0
        z_relativo = punto.z - z0

        landmarks_relativos.append([ x_relativo, y_relativo, z_relativo])

    # Normalización por distancia máxima
    distancia_maxima = max(
        math.sqrt(x**2 + y**2 + z**2)
        for x, y, z in landmarks_relativos
    )

    if distancia_maxima == 0:
        return None, "Error normalizacion"

    fila_landmarks = []

    for x, y, z in landmarks_relativos:
        fila_landmarks.extend([x / distancia_maxima, y / distancia_maxima, z / distancia_maxima])

    # Deben ser 63 características
    if len(fila_landmarks) != 63:
        return None, "Error dimensiones"

    prediccion = modelo_estatico.predict([fila_landmarks])[0]

    return str(prediccion), "OK"


# procesar videos
def procesar_video(ruta):

    global timestamp

    cap = cv2.VideoCapture(ruta)

    if not cap.isOpened():
        return None, "No se pudo abrir video"

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    fotogramas = []

    # Origen de movimiento
    xi = None
    yi = None
    zi = None

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        mano, tipo_mano = detectar_mano(frame, fps)

        if mano is None:
            continue

        muneca = mano[0]

        x0 = muneca.x
        y0 = muneca.y
        z0 = muneca.z

        # Primer frame válido = origen
        if xi is None:
            xi = x0
            yi = y0
            zi = z0

        landmarks_relativos = []

        for punto in mano:

            x_rel = punto.x - x0

            if tipo_mano == "Left":
                x_rel = -x_rel

            landmarks_relativos.append([ x_rel, punto.y - y0, punto.z - z0 ])

        dist_max = max( math.sqrt(x**2 + y**2 + z**2) for x, y, z in landmarks_relativos)

        if dist_max == 0:
            continue

        vector_frame = []

        # 63 valores de landmarks
        for x, y, z in landmarks_relativos:
            vector_frame.extend([ x / dist_max, y / dist_max,z / dist_max])

        # Movimiento global de la muñeca
        dx_total = x0 - xi
        dy_total = y0 - yi
        dz_total = z0 - zi

        if tipo_mano == "Left":
            dx_total = -dx_total

        # 63 + 3 = 66 características
        vector_frame.extend([
            dx_total,
            dy_total,
            dz_total
        ])

        fotogramas.append(vector_frame)

    cap.release()

    if len(fotogramas) == 0:
        return None, "No se detecto mano"

    # Convertir cualquier duración a 75 frames
    secuencia = convertir_a_75_frames(
        fotogramas
    )

    if secuencia is None:
        return None, "Error secuencia"

    # 75 × 66 = 4950
    if len(secuencia) != 4950:

        return (
            None,
            f"Dimensiones incorrectas: {len(secuencia)}"
        )

    prediccion = modelo_dinamico.predict(
        [secuencia]
    )[0]

    return str(prediccion), "OK"


# recorremos la carpeta donde estan las letras
resultados = []
print("INICIANDO EVALUACION")

for etiqueta_real in sorted(os.listdir(CARPETA_EVALUACION)):

    ruta_carpeta = os.path.join( CARPETA_EVALUACION, etiqueta_real)

    # Ignorar archivos que estén directamente
    # dentro de Evaluacion
    if not os.path.isdir(ruta_carpeta):
        continue

    print(f"\nEvaluando letra: {etiqueta_real}")

    archivos = sorted(os.listdir(ruta_carpeta))

    for archivo in archivos:
        ruta_archivo = os.path.join( ruta_carpeta, archivo)

        extension = os.path.splitext(archivo)[1].lower()
        prediccion = None
        estado = ""

        # IMagen ->modelo estarico
        if extension in EXTENSIONES_IMAGEN:
            tipo = "Estatica"
            prediccion, estado = procesar_imagen(ruta_archivo)

        #video -> modelo dinamico
        elif extension in EXTENSIONES_VIDEO:
            tipo = "Dinamica"
            prediccion, estado = procesar_video(ruta_archivo)

        else:
            continue

        #comparamos resultado
        if prediccion is None:
            resultado_final = "No evaluable"

        elif ( prediccion.upper()== etiqueta_real.upper()):
            resultado_final = "Correcta"

        else:
            resultado_final = "Incorrecta"

        print(
            f"  {archivo} -> "
            f"Real: {etiqueta_real} | "
            f"Predicha: {prediccion} | "
            f"{resultado_final}"
        )

        resultados.append({
            "archivo": archivo,
            "tipo": tipo,
            "etiqueta_real": etiqueta_real,
            "etiqueta_predicha": (
                prediccion
                if prediccion is not None
                else ""
            ),
            "resultado": resultado_final,
            "estado": estado
        })


# guardamos los datos obtenidos
df = pd.DataFrame(resultados)
df.to_csv( ARCHIVO_RESULTADOS, index=False, encoding="utf-8-sig")
print(f"\nResultados guardados en "f"{ARCHIVO_RESULTADOS}")

# resumimos resultados
resumen = []
for letra in sorted(df["etiqueta_real"].unique()):

    datos_letra = df[df["etiqueta_real"] == letra]
    total = len(datos_letra)

    correctas = (datos_letra["resultado"] == "Correcta").sum()
    incorrectas = ( datos_letra["resultado"] == "Incorrecta").sum()
    no_evaluables = ( datos_letra["resultado"]== "No evaluable").sum()

    porcentaje = ( correctas / total * 100 if total > 0 else 0)

    resumen.append({
        "letra": letra,
        "total_pruebas": total,
        "correctas": correctas,
        "incorrectas": incorrectas,
        "no_evaluables": no_evaluables,
        "porcentaje_correcto":
            round(porcentaje, 2)
    })

df_resumen = pd.DataFrame(resumen)
df_resumen.to_csv(ARCHIVO_RESUMEN, index=False,  encoding="utf-8-sig")

# resumen de la evaluacion
print("RESUMEN DE EVALUACIÓN")
print( df_resumen.to_string(index=False))
print( f"\nResumen guardado en "f"{ARCHIVO_RESUMEN}")
detector.close()
print("\nEvaluación terminada.")
