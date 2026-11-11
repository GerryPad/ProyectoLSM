package com.example.lsmreconocimiento

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.core.RunningMode
import java.io.Closeable

class HandLandmarkerHelper(context: Context) : Closeable {

    private val detector: HandLandmarker

    init {
        val opcionesBase = BaseOptions.builder()
            .setModelAssetPath("hand_landmarker.task")
            .build()

        val opciones = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(opcionesBase)
            .setRunningMode(RunningMode.IMAGE)
            .setNumHands(1)
            .build()

        detector = HandLandmarker.createFromOptions(
            context,
            opciones
        )
    }

    fun detectar(bitmap: Bitmap): HandLandmarkerResult {

        val imagen = BitmapImageBuilder(bitmap).build()

        return detector.detect(imagen)
    }

    override fun close() {
        detector.close()
    }
}
