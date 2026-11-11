package com.example.lsmreconocimiento

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.sqrt
import kotlin.math.floor
import kotlin.math.ceil

object DynamicLandmarkProcessor {

    private const val NUM_FOTOGRAMAS = 75
    private const val CARACTERISTICAS_FRAME = 66

    data class Muneca(
        val x: Float,
        val y: Float,
        val z: Float
    )

    // Obtener las coordenadas de la muñeca.
    fun obtenerMuneca(
        landmarks: List<NormalizedLandmark>
    ): Muneca {

        require(landmarks.size == 21)

        val punto = landmarks[0]

        return Muneca(
            punto.x(),
            punto.y(),
            punto.z()
        )
    }

    // Convertir un fotograma a 66 características.
    fun procesarFotograma(
        landmarks: List<NormalizedLandmark>,
        manoIzquierda: Boolean,
        origen: Muneca
    ): FloatArray? {

        if (landmarks.size != 21) {
            return null
        }

        val muneca = obtenerMuneca(landmarks)

        // Coordenadas relativas a la muñeca actual.
        val relativos = landmarks.map { punto ->

            var x = punto.x() - muneca.x

            if (manoIzquierda) {
                x = -x
            }

            val y = punto.y() - muneca.y
            val z = punto.z() - muneca.z

            floatArrayOf(x, y, z)
        }

        // Calcular distancia máxima.
        val distanciaMaxima = relativos.maxOf { punto ->

            val x = punto[0]
            val y = punto[1]
            val z = punto[2]

            sqrt(
                (x * x + y * y + z * z).toDouble()
            )
        }.toFloat()

        if (distanciaMaxima <= 0f) {
            return null
        }

        val vector = FloatArray(CARACTERISTICAS_FRAME)

        var indice = 0

        // Primeras 63 características.
        for (punto in relativos) {

            vector[indice++] =
                punto[0] / distanciaMaxima

            vector[indice++] =
                punto[1] / distanciaMaxima

            vector[indice++] =
                punto[2] / distanciaMaxima
        }

        // Últimas 3 características:
        // desplazamiento respecto al origen.
        var dx = muneca.x - origen.x

        if (manoIzquierda) {
            dx = -dx
        }

        val dy = muneca.y - origen.y
        val dz = muneca.z - origen.z

        vector[63] = dx
        vector[64] = dy
        vector[65] = dz

        return vector
    }

    // Convertir una secuencia variable a 75 fotogramas.
    fun convertirA75Fotogramas(
        fotogramas: List<FloatArray>
    ): FloatArray? {

        if (fotogramas.isEmpty()) {
            return null
        }

        if (fotogramas.any {
                it.size != CARACTERISTICAS_FRAME
            }) {
            return null
        }

        val resultado = FloatArray(
            NUM_FOTOGRAMAS * CARACTERISTICAS_FRAME
        )

        val cantidadOriginal = fotogramas.size

        for (i in 0 until NUM_FOTOGRAMAS) {

            val posicion =
                i.toDouble() *
                        (cantidadOriginal - 1) /
                        (NUM_FOTOGRAMAS - 1)

            val inferior = floor(posicion).toInt()
            val superior = ceil(posicion).toInt()
                .coerceAtMost(cantidadOriginal - 1)

            val peso = (posicion - inferior).toFloat()

            for (j in 0 until CARACTERISTICAS_FRAME) {

                val valorInferior =
                    fotogramas[inferior][j]

                val valorSuperior =
                    fotogramas[superior][j]

                val interpolado =
                    (1f - peso) * valorInferior +
                            peso * valorSuperior

                resultado[
                    i * CARACTERISTICAS_FRAME + j
                ] = interpolado
            }
        }

        return resultado
    }
}
