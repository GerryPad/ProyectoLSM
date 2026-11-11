package com.example.lsmreconocimiento

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log

import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView

import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.Alignment
import androidx.compose.ui.unit.dp
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner

import java.util.concurrent.Executors

@Composable
fun CameraScreen() {

    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    var letra by remember {
        mutableStateOf("Esperando mano...")
    }

    var puntuacion by remember {
        mutableFloatStateOf(0f)
    }

    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
    }

    val permisoConcedido =
        ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

    if (!permisoConcedido) {
        Text("Se necesita permiso de cámara")
        return
    }

    DisposableEffect(lifecycleOwner) {

        val executor = Executors.newSingleThreadExecutor()

        val cameraProviderFuture =
            ProcessCameraProvider.getInstance(context)

        var cameraProvider: ProcessCameraProvider? = null
        var activo = true
        var recursos: Pair<HandLandmarkerHelper?, LSMClassifier?>? = null

        val listener = Runnable {

            if (!activo) return@Runnable

            try {
                val provider = cameraProviderFuture.get()
                cameraProvider = provider

                val preview = Preview.Builder()
                    .build()
                    .also {
                        it.surfaceProvider =
                            previewView.surfaceProvider
                    }

                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(
                        ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
                    )
                    .setOutputImageFormat(
                        ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
                    )
                    .build()

                executor.execute {

                    try {

                        // Inicializar MediaPipe.
                        val detectorActivo = HandLandmarkerHelper(context)

                        // Inicializar TensorFlow Lite.
                        val clasificadorActivo = try {
                            LSMClassifier(context)
                        } catch (e: Exception) {
                            detectorActivo.close()
                            throw e
                        }

                        // Guardar referencias para liberar los recursos.
                        recursos = Pair(detectorActivo, clasificadorActivo)

                        // Configurar análisis de imágenes.
                        analysis.setAnalyzer(executor) { imageProxy ->

                            try {

                                // Convertir fotograma de CameraX a Bitmap.
                                val bitmap = convertirImagen(imageProxy)

                                // Detectar mano utilizando MediaPipe.
                                val resultado = detectorActivo.detectar(bitmap)

                                if (resultado.landmarks().isNotEmpty()) {

                                    // Obtener los 21 landmarks.
                                    val mano = resultado.landmarks()[0]

                                    // Identificar orientación de la mano.
                                    val tipoMano = resultado.handednesses()
                                        .firstOrNull()
                                        ?.firstOrNull()
                                        ?.categoryName()

                                    // Normalizar coordenadas.
                                    val caracteristicas =
                                        LandmarkProcessor.procesarEstatico(
                                            mano,
                                            tipoMano == "Left"
                                        )

                                    if (caracteristicas != null) {

                                        // Ejecutar modelo estático.
                                        val prediccion =
                                            clasificadorActivo.predecir(
                                                caracteristicas
                                            )

                                        // Actualizar interfaz.
                                        ContextCompat.getMainExecutor(context)
                                            .execute {

                                                if (activo) {
                                                    letra = prediccion.letra
                                                    puntuacion = prediccion.puntuacion
                                                }
                                            }
                                    }

                                } else {

                                    // No se detectó ninguna mano.
                                    ContextCompat.getMainExecutor(context)
                                        .execute {

                                            if (activo) {
                                                letra = "Mano no detectada"
                                                puntuacion = 0f
                                            }
                                        }
                                }

                            } catch (e: Exception) {

                                Log.e(
                                    "LSM",
                                    "Error durante el reconocimiento",
                                    e
                                )

                            } finally {

                                // Liberar fotograma para continuar capturando.
                                imageProxy.close()
                            }
                        }

                    } catch (e: Exception) {

                        Log.e(
                            "LSM",
                            "Error al inicializar los modelos",
                            e
                        )

                        ContextCompat.getMainExecutor(context)
                            .execute {

                                if (activo) {
                                    letra = "Error al cargar modelo"
                                }
                            }
                    }
                }

                provider.unbindAll()

                provider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_FRONT_CAMERA,
                    preview,
                    analysis
                )

            } catch (e: Exception) {
                Log.e("LSM", "Error iniciando cámara", e)
            }
        }

        cameraProviderFuture.addListener(
            listener,
            ContextCompat.getMainExecutor(context)
        )

        onDispose {

            activo = false

            // Detener el análisis de imágenes.
            cameraProvider?.unbindAll()

            // Liberar modelos después de las tareas pendientes.
            executor.execute {

                recursos?.first?.close()
                recursos?.second?.close()

                recursos = null
            }

            // Finalizar el executor cuando termine sus tareas.
            executor.shutdown()
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {

        AndroidView(
            factory = { previewView },
            modifier = Modifier.fillMaxSize()
        )

        Column(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(top = 55.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {

            Text(
                text = "Reconocimiento LSM",
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )

            Spacer(modifier = Modifier.height(12.dp))

            Text(
                text = "Letra: $letra",
                fontSize = 32.sp,
                fontWeight = FontWeight.Bold,
                color = Color.Green
            )

            Text(
                text = "Puntuación: %.3f".format(puntuacion),
                fontSize = 16.sp,
                color = Color.White
            )
        }
    }
}

private fun convertirImagen(
    imageProxy: androidx.camera.core.ImageProxy
): Bitmap {

    val plane = imageProxy.planes[0]
    val buffer = plane.buffer

    buffer.rewind()

    val pixelStride = plane.pixelStride
    val rowStride = plane.rowStride

    require(pixelStride == 4) {
        "Formato RGBA inesperado"
    }

    val ancho = imageProxy.width
    val alto = imageProxy.height

    // El stride puede incluir bytes de relleno.
    val anchoBuffer = rowStride / pixelStride

    val bitmapCompleto = Bitmap.createBitmap(
        anchoBuffer,
        alto,
        Bitmap.Config.ARGB_8888
    )

    bitmapCompleto.copyPixelsFromBuffer(buffer)

    val bitmap = Bitmap.createBitmap(
        bitmapCompleto,
        0,
        0,
        ancho,
        alto
    )

    val matriz = Matrix().apply {
        postRotate(
            imageProxy.imageInfo.rotationDegrees.toFloat()
        )

        // Reproducir el efecto espejo de cv2.flip(frame, 1).
        postScale(-1f, 1f)
    }

    return Bitmap.createBitmap(
        bitmap,
        0,
        0,
        bitmap.width,
        bitmap.height,
        matriz,
        true
    )
}