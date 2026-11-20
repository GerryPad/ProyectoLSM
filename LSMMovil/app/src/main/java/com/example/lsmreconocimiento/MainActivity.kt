package com.example.lsmreconocimiento

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle

import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts

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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

import androidx.core.content.ContextCompat

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            LSMApp()
        }
    }
}
enum class PantallaLSM {
    MENU,
    PRACTICA,
    DELETREO
}

@Composable
fun LSMApp() {
    val context = LocalContext.current
    var pantallaActual by remember {
        mutableStateOf(PantallaLSM.MENU)
    }
    var pantallaPendiente by remember {
        mutableStateOf(PantallaLSM.PRACTICA)
    }
    var permisoConcedido by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        )
    }

    val solicitarPermiso =
        rememberLauncherForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { concedido ->
            permisoConcedido = concedido
            if (concedido) {
                pantallaActual = pantallaPendiente
            }
        }

    fun abrirPantalla(pantalla: PantallaLSM) {
        if (permisoConcedido) {
            pantallaActual = pantalla
        } else {
            pantallaPendiente = pantalla

            solicitarPermiso.launch(
                Manifest.permission.CAMERA
            )
        }
    }

    when (pantallaActual) {
        PantallaLSM.MENU -> {
            MenuScreen(
                onPractica = {
                    abrirPantalla(PantallaLSM.PRACTICA)
                },
                onDeletreo = {
                    abrirPantalla(PantallaLSM.DELETREO)
                }
            )
        }

        PantallaLSM.PRACTICA -> {
            CameraScreen(
                onRegresar = {
                    pantallaActual = PantallaLSM.MENU
                }
            )
        }

        PantallaLSM.DELETREO -> {
            DeletreoScreen(
                onRegresar = {
                    pantallaActual = PantallaLSM.MENU
                }
            )
        }
    }
}

@Composable
fun MenuScreen(
    onPractica: () -> Unit,
    onDeletreo: () -> Unit
) {

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF101820))
            .padding(24.dp),

        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {

        Text(
            text = "Reconocimiento LSM",
            fontSize = 30.sp,
            fontWeight = FontWeight.Bold,
            color = Color.White
        )

        Spacer(
            modifier = Modifier.height(16.dp)
        )

        Text(
            text = "Lengua de Señas Mexicana",
            fontSize = 18.sp,
            color = Color.LightGray
        )

        Spacer(
            modifier = Modifier.height(60.dp)
        )

        Button(
            onClick = onPractica,

            modifier = Modifier
                .fillMaxWidth()
                .height(65.dp),

            shape = RoundedCornerShape(16.dp),

            colors = ButtonDefaults.buttonColors(
                containerColor = Color(0xFF673AB7)
            )
        ) {

            Text(
                text = "Práctica",
                fontSize = 22.sp
            )
        }

        Spacer(
            modifier = Modifier.height(24.dp)
        )

        Button(
            onClick = onDeletreo,

            modifier = Modifier
                .fillMaxWidth()
                .height(65.dp),

            shape = RoundedCornerShape(16.dp),

            colors = ButtonDefaults.buttonColors(
                containerColor = Color(0xFF1976D2)
            )
        ) {

            Text(
                text = "Deletreo",
                fontSize = 22.sp
            )
        }
    }
}

