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
import androidx.compose.foundation.background
import androidx.compose.foundation.shape.RoundedCornerShape
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

    var estado by remember {
        mutableStateOf("ESPERANDO")
    }

    var landmarksActuales by remember {
        mutableStateOf<
                List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>
                >(emptyList())
    }

    var anchoImagen by remember {
        mutableIntStateOf(0)
    }

    var altoImagen by remember {
        mutableIntStateOf(0)
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
        var recursos: Triple<
                HandLandmarkerHelper,
                LSMClassifier,
                DynamicLSMClassifier
                >? = null

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
                        // Inicializar el modelo dinámico.
                        val clasificadorDinamico = try {
                            DynamicLSMClassifier(context)
                        } catch (e: Exception) {
                            clasificadorActivo.close()
                            detectorActivo.close()
                            throw e
                        }

                        // Crear la máquina de estados.
                        val reconocedorDinamico = DynamicRecognizer(
                            clasificadorDinamico
                        )

                        // Guardar referencias para liberar los recursos.
                        recursos = Triple(
                            detectorActivo,
                            clasificadorActivo,
                            clasificadorDinamico
                        )

                        // Configurar análisis de imágenes.
                        analysis.setAnalyzer(executor) { imageProxy ->

                            try {

                                // ==================================
                                // 1. CONVERTIR IMAGEN
                                // ==================================

                                val bitmap = convertirImagen(imageProxy)

                                // ==================================
                                // 2. DETECTAR MANO
                                // ==================================

                                val resultado = detectorActivo.detectar(bitmap)

                                if (resultado.landmarks().isNotEmpty()) {

                                    val mano = resultado.landmarks()[0]

                                    val tipoMano = resultado.handednesses()
                                        .firstOrNull()
                                        ?.firstOrNull()
                                        ?.categoryName()

                                    val manoIzquierda = tipoMano == "Left"

                                    // ==================================
                                    // 3. ACTUALIZAR LANDMARKS VISUALES
                                    // ==================================

                                    val puntosDetectados = mano.toList()

                                    val anchoDetectado = bitmap.width
                                    val altoDetectado = bitmap.height

                                    ContextCompat.getMainExecutor(context).execute {

                                        if (activo) {

                                            landmarksActuales = puntosDetectados

                                            anchoImagen = anchoDetectado
                                            altoImagen = altoDetectado
                                        }
                                    }

                                    // ==================================
                                    // 4. ANALIZAR MOVIMIENTO
                                    // ==================================

                                    val resultadoDinamico =
                                        reconocedorDinamico.procesar(
                                            landmarks = mano,
                                            manoIzquierda = manoIzquierda
                                        )

                                    // ==================================
                                    // 5. DECIDIR TIPO DE RECONOCIMIENTO
                                    // ==================================

                                    when (resultadoDinamico.estado) {

                                        DynamicRecognizer.Estado.ESPERANDO -> {

                                            // Reconocimiento estático.

                                            val caracteristicas =
                                                LandmarkProcessor.procesarEstatico(
                                                    mano,
                                                    manoIzquierda
                                                )

                                            if (caracteristicas != null) {

                                                val prediccion =
                                                    clasificadorActivo.predecir(
                                                        caracteristicas
                                                    )

                                                ContextCompat.getMainExecutor(context)
                                                    .execute {

                                                        if (activo) {

                                                            letra = prediccion.letra

                                                            puntuacion =
                                                                prediccion.puntuacion

                                                            estado = "DETECTADO"
                                                        }
                                                    }
                                            }
                                        }

                                        DynamicRecognizer.Estado.CAPTURANDO -> {

                                            // Se está realizando una seña dinámica.
                                            // No ejecutamos el modelo estático.

                                            ContextCompat.getMainExecutor(context)
                                                .execute {

                                                    if (activo) {

                                                        estado = "CAPTURANDO"

                                                        letra = "..."

                                                        puntuacion = 0f
                                                    }
                                                }
                                        }

                                        DynamicRecognizer.Estado.PAUSA -> {

                                            // La captura terminó.
                                            // Mostrar resultado si existe una predicción.

                                            val prediccion =
                                                resultadoDinamico.prediccion

                                            if (prediccion != null) {

                                                ContextCompat.getMainExecutor(context)
                                                    .execute {

                                                        if (activo) {

                                                            letra = prediccion.letra

                                                            puntuacion =
                                                                prediccion.puntuacion

                                                            estado = "DETECTADO"
                                                        }
                                                    }
                                            }
                                        }
                                    }

                                } else {

                                    // ==================================
                                    // 6. MANO NO DETECTADA
                                    // ==================================

                                    val resultadoDinamico =
                                        reconocedorDinamico.procesar(
                                            landmarks = null,
                                            manoIzquierda = false
                                        )

                                    ContextCompat.getMainExecutor(context)
                                        .execute {

                                            if (activo) {

                                                landmarksActuales = emptyList()

                                                if (resultadoDinamico.prediccion != null) {

                                                    // Se completó una seña al retirar la mano.
                                                    letra =
                                                        resultadoDinamico.prediccion.letra

                                                    puntuacion =
                                                        resultadoDinamico.prediccion.puntuacion

                                                    estado = "DETECTADO"

                                                } else {

                                                    letra = "Mano no detectada"

                                                    puntuacion = 0f

                                                    estado = "SIN_MANO"
                                                }
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
                                    puntuacion = 0f
                                    estado = "ERROR"
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
                recursos?.third?.close()

                recursos = null
            }

            // Finalizar el executor cuando termine sus tareas.
            executor.shutdown()
        }
    }

    Box(
        modifier = Modifier.fillMaxSize()
    ) {

        // ==================================
        // VISTA DE LA CÁMARA
        // ==================================

        AndroidView(
            factory = { previewView },
            modifier = Modifier.fillMaxSize()
        )

        // VISUALIZACIÓN DE LANDMARKS
        HandLandmarksOverlay(
            landmarks = landmarksActuales,
            imageWidth = anchoImagen,
            imageHeight = altoImagen,
            modifier = Modifier.fillMaxSize()
        )

        // ==================================
        // COLORES SEGÚN EL ESTADO
        // ==================================

        val colorEstado = when (estado) {
            "DETECTADO" -> Color(0xFF4CAF50)
            "CAPTURANDO" -> Color(0xFF03A9F4)
            "SIN_MANO" -> Color(0xFFFF9800)
            "ERROR" -> Color(0xFFF44336)
            else -> Color(0xFFFFEB3B)
        }

        val mensajeEstado = when (estado) {
            "DETECTADO" -> "Mano detectada"
            "CAPTURANDO" -> "Capturando seña dinámica..."
            "SIN_MANO" -> "Mano no detectada"
            "ERROR" -> "Error de reconocimiento"
            else -> "Esperando detección..."
        }

        // ==================================
        // PANEL DE PREDICCIÓN
        // ==================================

        Column(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .fillMaxWidth()
                .padding(
                    horizontal = 16.dp,
                    vertical = 40.dp
                )
                .background(
                    color = Color(0xFF17212B).copy(alpha = 0.88f),
                    shape = RoundedCornerShape(20.dp)
                )
                .padding(20.dp),

            horizontalAlignment = Alignment.CenterHorizontally
        ) {

            Text(
                text = "Reconocimiento LSM",
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White
            )

            Spacer(
                modifier = Modifier.height(12.dp)
            )

            // Estado actual
            Text(
                text = mensajeEstado,
                fontSize = 18.sp,
                fontWeight = FontWeight.Medium,
                color = colorEstado
            )

            Spacer(
                modifier = Modifier.height(16.dp)
            )

            // Letra reconocida
            Text(
                text = if (estado == "DETECTADO") {
                    letra
                } else {
                    "-"
                },
                fontSize = 64.sp,
                fontWeight = FontWeight.Bold,
                color = colorEstado
            )

            Spacer(
                modifier = Modifier.height(8.dp)
            )

            // Puntuación del modelo
            Text(
                text = if (estado == "DETECTADO") {
                    "Puntuación: %.3f".format(puntuacion)
                } else {
                    "Puntuación: --"
                },
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