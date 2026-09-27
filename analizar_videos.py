import os
import cv2
import numpy as np

carpeta_videos = (
    "dataset/MSL-dynamic-signs-frontal-view/"
    "MSL-dynamic-signs/train"
)

# Todos los videos juntos
conteo_frames = []

# Frames separados por letra
frames_por_letra = {}

print("Analizando la duración de los videos dinámicos...\n")


if os.path.exists(carpeta_videos):

    for archivo in os.listdir(carpeta_videos):

        if archivo.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):

            # -------------------------------------------------
            # 1. OBTENER LA LETRA DEL NOMBRE
            # -------------------------------------------------

            # Ejemplo:
            # S1-J-frontal-1.mp4
            #
            # split('-') ->
            # ["S1", "J", "frontal", "1.mp4"]

            partes = archivo.split('-')

            if len(partes) < 2:
                print("Nombre no reconocido:", archivo)
                continue

            letra = partes[1].upper()


            # -------------------------------------------------
            # 2. ABRIR VIDEO
            # -------------------------------------------------

            ruta_video = os.path.join(
                carpeta_videos,
                archivo
            )

            cap = cv2.VideoCapture(ruta_video)

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            fps = cap.get(cv2.CAP_PROP_FPS)

            cap.release()


            # -------------------------------------------------
            # 3. GUARDAR RESULTADO
            # -------------------------------------------------

            if total_frames > 0:

                conteo_frames.append(total_frames)

                if letra not in frames_por_letra:
                    frames_por_letra[letra] = []

                frames_por_letra[letra].append(
                    total_frames
                )


    # -------------------------------------------------
    # 4. RESULTADOS GENERALES
    # -------------------------------------------------

    if len(conteo_frames) > 0:

        datos = np.array(conteo_frames)

        print("=" * 50)
        print("RESULTADOS GENERALES")
        print("=" * 50)

        print(
            "Total de videos:",
            len(datos)
        )

        print(
            f"Promedio: {np.mean(datos):.1f} frames"
        )

        print(
            f"Mediana: {np.median(datos):.1f} frames"
        )

        print(
            f"Mínimo: {np.min(datos)} frames"
        )

        print(
            f"Máximo: {np.max(datos)} frames"
        )

        print(
            f"Percentil 25: {np.percentile(datos, 25):.1f}"
        )

        print(
            f"Percentil 50: {np.percentile(datos, 50):.1f}"
        )

        print(
            f"Percentil 75: {np.percentile(datos, 75):.1f}"
        )

        print(
            f"Percentil 90: {np.percentile(datos, 90):.1f}"
        )


        # -------------------------------------------------
        # 5. RESULTADOS POR LETRA
        # -------------------------------------------------

        print("\n")
        print("=" * 50)
        print("RESULTADOS POR LETRA")
        print("=" * 50)

        for letra in sorted(frames_por_letra):

            valores = np.array(
                frames_por_letra[letra]
            )

            print(f"\nLetra: {letra}")

            print(
                f"  Videos: {len(valores)}"
            )

            print(
                f"  Promedio: {np.mean(valores):.1f}"
            )

            print(
                f"  Mediana: {np.median(valores):.1f}"
            )

            print(
                f"  Mínimo: {np.min(valores)}"
            )

            print(
                f"  Máximo: {np.max(valores)}"
            )

            print(
                f"  P25: {np.percentile(valores, 25):.1f}"
            )

            print(
                f"  P75: {np.percentile(valores, 75):.1f}"
            )

    else:

        print(
            "No se encontraron videos válidos."
        )

else:

    print(
        f"La ruta '{carpeta_videos}' no existe."
    )