package com.example.lsmreconocimiento

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.sqrt

object LandmarkProcessor {

    fun procesarEstatico(
        landmarks: List<NormalizedLandmark>,
        manoIzquierda: Boolean
    ): FloatArray? {

        // Necesitamos exactamente 21 landmarks.
        if (landmarks.size != 21) {
            return null
        }

        // Landmark 0: muñeca.
        val muneca = landmarks[0]

        val x0 = muneca.x()
        val y0 = muneca.y()
        val z0 = muneca.z()

        // Coordenadas relativas a la muñeca.
        val relativos = landmarks.map { punto ->

            var x = punto.x() - x0

            if (manoIzquierda) {
                x = -x
            }

            val y = punto.y() - y0
            val z = punto.z() - z0

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

        // Vector de 63 características normalizadas.
        val caracteristicas = FloatArray(63)

        var indice = 0

        for (punto in relativos) {

            caracteristicas[indice++] =
                punto[0] / distanciaMaxima

            caracteristicas[indice++] =
                punto[1] / distanciaMaxima

            caracteristicas[indice++] =
                punto[2] / distanciaMaxima
        }

        return caracteristicas
    }
}
