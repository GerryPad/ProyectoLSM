package com.example.lsmreconocimiento

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark

@Composable
fun HandLandmarksOverlay(
    landmarks: List<NormalizedLandmark>,
    imageWidth: Int,
    imageHeight: Int,
    modifier: Modifier = Modifier
) {

    // Conexiones entre los 21 puntos.
    val conexiones = listOf(

        // Pulgar
        0 to 1,
        1 to 2,
        2 to 3,
        3 to 4,

        // Índice
        0 to 5,
        5 to 6,
        6 to 7,
        7 to 8,

        // Medio
        5 to 9,
        9 to 10,
        10 to 11,
        11 to 12,

        // Anular
        9 to 13,
        13 to 14,
        14 to 15,
        15 to 16,

        // Meñique
        13 to 17,
        17 to 18,
        18 to 19,
        19 to 20,

        // Palma
        0 to 17
    )

    Canvas(
        modifier = modifier.fillMaxSize()
    ) {

        if (
            landmarks.size != 21 ||
            imageWidth <= 0 ||
            imageHeight <= 0
        ) {
            return@Canvas
        }

        // Equivalente a ContentScale.Crop:
        // conserva la proporción y llena la pantalla.
        val escala = maxOf(
            size.width / imageWidth,
            size.height / imageHeight
        )

        val anchoEscalado = imageWidth * escala
        val altoEscalado = imageHeight * escala

        val desplazamientoX =
            (size.width - anchoEscalado) / 2f

        val desplazamientoY =
            (size.height - altoEscalado) / 2f

        fun posicion(indice: Int): Offset {

            val punto = landmarks[indice]

            return Offset(
                x = punto.x() * anchoEscalado + desplazamientoX,
                y = punto.y() * altoEscalado + desplazamientoY
            )
        }

        // =================================
        // DIBUJAR CONEXIONES
        // =================================

        conexiones.forEach { (inicio, fin) ->

            drawLine(
                color = Color(0xFF0F248F),
                start = posicion(inicio),
                end = posicion(fin),
                strokeWidth = 5f
            )
        }

        // =================================
        // DIBUJAR LANDMARKS
        // =================================

        landmarks.indices.forEach { indice ->

            val punto = posicion(indice)

            // Contorno oscuro para mejorar visibilidad.
            drawCircle(
                color = Color.Black,
                radius = 9f,
                center = punto
            )

            // Punto principal.
            drawCircle(
                color = Color(0xFF17D918),
                radius = 6f,
                center = punto
            )
        }
    }
}
