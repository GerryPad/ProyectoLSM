package com.example.lsmreconocimiento

import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.DataType
import java.io.Closeable

class LSMClassifier(context: Context) : Closeable {

    private val clasesEstaticas = listOf(
        "A", "B", "C", "D", "E", "F", "G",
        "H", "I", "L", "M", "N", "O",
        "P", "R", "S", "T", "U", "V",
        "W", "Y"
    )

    private val interpreter: Interpreter

    init {
        // Cargar el modelo desde assets.
        val modelo = context.assets.open(
            "lsm_modelo_estatico.tflite"
        ).use { it.readBytes() }

        interpreter = Interpreter(
            java.nio.ByteBuffer.allocateDirect(modelo.size)
                .order(java.nio.ByteOrder.nativeOrder())
                .apply {
                    put(modelo)
                    rewind()
                }
        )

        // Verificar las dimensiones y tipos del modelo.
        val entrada = interpreter.getInputTensor(0)
        val salida = interpreter.getOutputTensor(0)

        require(entrada.shape().contentEquals(intArrayOf(1, 63))) {
            "Entrada inesperada: ${entrada.shape().contentToString()}"
        }

        require(entrada.dataType() == DataType.FLOAT32) {
            "El modelo no utiliza entradas FLOAT32"
        }

        require(salida.shape().contentEquals(intArrayOf(1, 21))) {
            "Salida inesperada: ${salida.shape().contentToString()}"
        }

        require(salida.dataType() == DataType.FLOAT32) {
            "El modelo no utiliza salidas FLOAT32"
        }
    }

    data class Prediccion(
        val letra: String,
        val puntuacion: Float
    )

    fun predecir(caracteristicas: FloatArray): Prediccion {

        require(caracteristicas.size == 63) {
            "Se esperaban 63 características"
        }

        val entrada = arrayOf(caracteristicas)
        val salida = Array(1) { FloatArray(21) }

        interpreter.run(entrada, salida)

        val puntuaciones = salida[0]

        val indice = puntuaciones.indices.maxByOrNull {
            puntuaciones[it]
        } ?: error("No se obtuvo ninguna predicción")

        return Prediccion(
            letra = clasesEstaticas[indice],
            puntuacion = puntuaciones[indice]
        )
    }

    override fun close() {
        interpreter.close()
    }
}
