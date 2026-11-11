package com.example.lsmreconocimiento

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.sqrt

class DynamicRecognizer(
    private val clasificador: DynamicLSMClassifier
) {

    companion object {
        private const val UMBRAL_MOVIMIENTO = 0.025f
        private const val FRAMES_MINIMOS = 15
        private const val FRAMES_MAXIMOS = 60
        private const val FRAMES_REPOSO = 8

        private const val PAUSA_MS = 1500L
    }

    enum class Estado {
        ESPERANDO,
        CAPTURANDO,
        PAUSA
    }
    data class Resultado(
        val estado: Estado,
        val prediccion: DynamicLSMClassifier.PrediccionDinamica? = null
    )
    private var estado = Estado.ESPERANDO
    private var munecaAnterior:
            DynamicLandmarkProcessor.Muneca? = null
    private var origenCaptura:
            DynamicLandmarkProcessor.Muneca? = null
    private val fotogramas = mutableListOf<FloatArray>()
    private var framesQuietos = 0
    private var inicioPausa = 0L
    private var ultimaPrediccion:
            DynamicLSMClassifier.PrediccionDinamica? = null

    fun procesar(
        landmarks: List<NormalizedLandmark>?,
        manoIzquierda: Boolean,
        tiempoActual: Long = System.nanoTime() / 1_000_000L
    ): Resultado {

        // Si desaparece la mano, finalizar la captura.
        if (landmarks == null || landmarks.size != 21) {

            // Mantener el resultado durante la pausa.
            if (estado == Estado.PAUSA) {

                if (tiempoActual - inicioPausa < PAUSA_MS) {
                    return Resultado(
                        estado = Estado.PAUSA,
                        prediccion = ultimaPrediccion
                    )
                }

                reiniciar()
                return Resultado(Estado.ESPERANDO)
            }

            // Si la mano desaparece durante la captura,
            // procesar los fotogramas disponibles.
            if (estado == Estado.CAPTURANDO) {

                val prediccion = if (
                    fotogramas.size >= FRAMES_MINIMOS
                ) {
                    predecirSecuencia()
                } else {
                    null
                }

                if (prediccion != null) {

                    ultimaPrediccion = prediccion

                    estado = Estado.PAUSA
                    inicioPausa = tiempoActual

                    fotogramas.clear()
                    framesQuietos = 0
                    munecaAnterior = null

                    return Resultado(
                        estado = Estado.PAUSA,
                        prediccion = prediccion
                    )
                }
            }

            reiniciar()

            return Resultado(Estado.ESPERANDO)
        }

        val muneca =
            DynamicLandmarkProcessor.obtenerMuneca(landmarks)

        // Calcular desplazamiento de la muñeca
        // respecto al fotograma anterior.
        val movimiento = munecaAnterior?.let { anterior ->

            val dx = muneca.x - anterior.x
            val dy = muneca.y - anterior.y
            val dz = muneca.z - anterior.z

            sqrt(
                (dx * dx + dy * dy + dz * dz).toDouble()
            ).toFloat()

        } ?: 0f

        munecaAnterior = muneca

        when (estado) {

            Estado.ESPERANDO -> {

                if (movimiento >= UMBRAL_MOVIMIENTO) {

                    fotogramas.clear()
                    framesQuietos = 0
                    origenCaptura = null

                    estado = Estado.CAPTURANDO
                }
            }

            Estado.CAPTURANDO -> {

                if (origenCaptura == null) {
                    origenCaptura = muneca
                }

                val origen = origenCaptura!!

                val vector =
                    DynamicLandmarkProcessor.procesarFotograma(
                        landmarks = landmarks,
                        manoIzquierda = manoIzquierda,
                        origen = origen
                    )

                if (vector != null) {
                    fotogramas.add(vector)
                }

                if (movimiento < UMBRAL_MOVIMIENTO) {
                    framesQuietos++
                } else {
                    framesQuietos = 0
                }

                val terminar =
                    framesQuietos >= FRAMES_REPOSO ||
                            fotogramas.size >= FRAMES_MAXIMOS

                if (terminar) {

                    val prediccion = if (
                        fotogramas.size >= FRAMES_MINIMOS
                    ) {
                        predecirSecuencia()
                    } else {
                        null
                    }
                    ultimaPrediccion = prediccion

                    fotogramas.clear()
                    framesQuietos = 0

                    estado = Estado.PAUSA
                    inicioPausa = tiempoActual

                    return Resultado(
                        estado = Estado.PAUSA,
                        prediccion = prediccion
                    )
                }
            }

            Estado.PAUSA -> {

                if (tiempoActual - inicioPausa >= PAUSA_MS) {

                    estado = Estado.ESPERANDO
                    munecaAnterior = muneca
                }
                ultimaPrediccion = null
            }
        }

        return Resultado(
            estado = estado,
            prediccion = if (estado == Estado.PAUSA) {
                ultimaPrediccion
            } else {
                null
            }
        )
    }

    private fun predecirSecuencia():
            DynamicLSMClassifier.PrediccionDinamica? {

        val caracteristicas =
            DynamicLandmarkProcessor.convertirA75Fotogramas(
                fotogramas
            ) ?: return null

        return clasificador.predecir(caracteristicas)
    }

    private fun reiniciar() {

        estado = Estado.ESPERANDO

        munecaAnterior = null
        origenCaptura = null
        ultimaPrediccion = null
        fotogramas.clear()
        framesQuietos = 0
        inicioPausa = 0L
    }
}
