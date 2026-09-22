import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Usamos model_complexity=0 para aligerar el procesamiento de la red neuronal
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,              # Limitamos a 1 sola mano para acelerar el cálculo
    model_complexity=0,           # Modelo ligero (0 en lugar de 1)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

camara = cv2.VideoCapture(0)

while True:
    ret, frame = camara.read()
    
    if ret:
        # REDIMENSIONAR: Achicamos el frame a 640x480 para que la computadora no sufra
        frame = cv2.resize(frame, (640, 480))
        
        # Opcional: Voltear horizontalmente para que actúe como espejo de forma natural
        frame = cv2.flip(frame, 1)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = hands.process(frame_rgb)
        
        if resultados.multi_hand_landmarks:
            for hand_landmarks in resultados.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )
                
        cv2.imshow("Mi camara optimizada", frame)
        
    if cv2.waitKey(1) == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()