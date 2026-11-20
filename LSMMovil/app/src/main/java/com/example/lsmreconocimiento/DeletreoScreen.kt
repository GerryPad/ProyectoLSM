package com.example.lsmreconocimiento

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape

import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Text

import androidx.compose.runtime.*

import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun DeletreoScreen(
    onRegresar: () -> Unit
) {
    // VARIABLES DE ESTADO
    var letraPredicha by remember {
        mutableStateOf<String?>(null)
    }

    var puntuacion by remember {
        mutableFloatStateOf(0f)
    }

    val letrasConfirmadas = remember {
        mutableStateListOf<String>()
    }

    var finalizado by remember {
        mutableStateOf(false)
    }

    // Variable temporal para probar la interfaz.
    // Se eliminará al conectar el reconocimiento.
    var indicePrueba by remember {
        mutableIntStateOf(0)
    }

    val letrasPrueba = listOf("H", "O", "L", "A")

    val palabra = letrasConfirmadas.joinToString("")

    // INTERFAZ PRINCIPAL
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF101820))
    ) {

        // ÁREA CENTRAL
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(
                    top = 260.dp,
                    bottom = 220.dp
                ),

            contentAlignment = Alignment.Center
        ) {

            Text(
                text = "Vista de cámara\n(Próxima etapa)",
                color = Color.LightGray,
                fontSize = 20.sp,
                textAlign = TextAlign.Center
            )
        }

        // BOTÓN REGRESAR
        Button(
            onClick = onRegresar,

            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(16.dp)
        ) {

            Text("← Regresar")
        }

        // PANEL SUPERIOR
        Column(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .fillMaxWidth()
                .padding(
                    start = 16.dp,
                    end = 16.dp,
                    top = 90.dp
                )
                .background(
                    color = Color(0xFF17212B),
                    shape = RoundedCornerShape(20.dp)
                )
                .padding(16.dp),

            horizontalAlignment = Alignment.CenterHorizontally
        ) {

            Text(
                text = "Deletreo LSM",
                color = Color.White,
                fontSize = 24.sp,
                fontWeight = FontWeight.Bold
            )

            Spacer(
                modifier = Modifier.height(12.dp)
            )

            Text(
                text = if (finalizado) {
                    "Palabra completada"
                } else {
                    "Letra detectada"
                },
                color = Color.LightGray,
                fontSize = 18.sp
            )

            Spacer(
                modifier = Modifier.height(12.dp)
            )

            Text(
                text = when {
                    finalizado -> palabra
                    letraPredicha != null -> letraPredicha!!
                    else -> "-"
                },

                color = Color(0xFF4CAF50),
                fontSize = if (finalizado) 40.sp else 64.sp,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center
            )

            if (!finalizado) {

                Spacer(
                    modifier = Modifier.height(8.dp)
                )

                Text(
                    text = if (letraPredicha != null) {
                        "Puntuación: %.3f".format(puntuacion)
                    } else {
                        "Esperando predicción..."
                    },

                    color = Color.White,
                    fontSize = 16.sp
                )
            }
        }

        // PANEL INFERIOR
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .padding(16.dp)
                .background(
                    color = Color(0xFF17212B),
                    shape = RoundedCornerShape(20.dp)
                )
                .padding(16.dp),

            horizontalAlignment = Alignment.CenterHorizontally
        ) {

            Text(
                text = "Palabra: $palabra",
                color = Color.White,
                fontSize = 22.sp,
                fontWeight = FontWeight.Bold
            )

            Spacer(
                modifier = Modifier.height(16.dp)
            )

            if (!finalizado) {

                // BOTÓN TEMPORAL DE PRUEBA
                Button(
                    onClick = {

                        letraPredicha = letrasPrueba[
                            indicePrueba % letrasPrueba.size
                        ]

                        puntuacion = 0.95f
                    },

                    enabled = letraPredicha == null
                ) {

                    Text("Simular predicción")
                }

                Spacer(
                    modifier = Modifier.height(12.dp)
                )

                Row(
                    modifier = Modifier.fillMaxWidth(),

                    horizontalArrangement = Arrangement.spacedBy(16.dp)
                ) {

                    // CONFIRMAR LETRA
                    Button(
                        onClick = {

                            letraPredicha?.let { letra ->

                                letrasConfirmadas.add(letra)

                                indicePrueba++

                                letraPredicha = null

                                puntuacion = 0f
                            }
                        },

                        enabled = letraPredicha != null,

                        modifier = Modifier.weight(1f),

                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFF388E3C)
                        )
                    ) {

                        Text(
                            text = "✔",
                            fontSize = 26.sp
                        )
                    }


                    // RECHAZAR LETRA
                    Button(
                        onClick = {

                            letraPredicha = null

                            puntuacion = 0f
                        },

                        enabled = letraPredicha != null,

                        modifier = Modifier.weight(1f),

                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFFD32F2F)
                        )
                    ) {

                        Text(
                            text = "✕",
                            fontSize = 26.sp
                        )
                    }
                }

                Spacer(
                    modifier = Modifier.height(12.dp)
                )


                // FINALIZAR DELETREO
                Button(
                    onClick = {
                        finalizado = true
                    },

                    enabled = letrasConfirmadas.isNotEmpty()
                            && letraPredicha == null,

                    modifier = Modifier.fillMaxWidth(),

                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFF1976D2)
                    )
                ) {

                    Text(
                        text = "Finish",
                        fontSize = 20.sp
                    )
                }

            } else {

                // NUEVA PALABRA
                Button(
                    onClick = {

                        letrasConfirmadas.clear()

                        letraPredicha = null

                        puntuacion = 0f

                        finalizado = false

                        indicePrueba = 0
                    },

                    modifier = Modifier.fillMaxWidth(),

                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFF673AB7)
                    )
                ) {

                    Text(
                        text = "Nueva palabra",
                        fontSize = 20.sp
                    )
                }
            }
        }
    }
}