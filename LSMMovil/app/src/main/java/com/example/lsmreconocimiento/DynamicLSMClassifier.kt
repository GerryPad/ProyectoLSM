package com.example.lsmreconocimiento

import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.DataType
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder

class DynamicLSMClassifier(
    context: Context
) : Closeable {

    private val clasesDinamicas = listOf(
        "J", "K", "Q", "X", "Z", "Ñ"
    )

    private val interpreter: Interpreter

    init {

        // Cargar modelo dinámico desde assets.
        val modelo = context.assets.open(
            "lsm_modelo_dinamico.tflite"
        ).use {
            it.readBytes()
        }

        val buffer = ByteBuffer.allocateDirect(
            modelo.size
        ).order(ByteOrder.nativeOrder())

        buffer.put(modelo)
        buffer.rewind()

        interpreter = Interpreter(buffer)

        // Comprobar dimensiones.
        val entrada = interpreter.getInputTensor(0)
        val salida = interpreter.getOutputTensor(0)

        require(
            entrada.shape().contentEquals(
                intArrayOf(1, 4950)
            )
        ) {
            "Dimensiones de entrada incorrectas: " +
                    entrada.shape().contentToString()
        }

        require(
            salida.shape().contentEquals(
                intArrayOf(1, 6)
            )
        ) {
            "Dimensiones de salida incorrectas: " +
                    salida.shape().contentToString()
        }

        require(entrada.dataType() == DataType.FLOAT32) {
            "El modelo requiere entrada FLOAT32"
        }

        require(salida.dataType() == DataType.FLOAT32) {
            "El modelo requiere salida FLOAT32"
        }
    }

    data class PrediccionDinamica(
        val letra: String,
        val puntuacion: Float
    )

    fun predecir(
        caracteristicas: FloatArray
    ): PrediccionDinamica {

        require(caracteristicas.size == 4950) {
            "Se esperaban 4950 características"
        }

        val entrada = arrayOf(caracteristicas)

        val salida = Array(1) {
            FloatArray(6)
        }

        interpreter.run(entrada, salida)

        val puntuaciones = salida[0]

        val indice = puntuaciones.indices.maxByOrNull {
            puntuaciones[it]
        } ?: error("No se obtuvo ninguna predicción")

        return PrediccionDinamica(
            letra = clasesDinamicas[indice],
            puntuacion = puntuaciones[indice]
        )
    }

    override fun close() {
        interpreter.close()
    }
}
