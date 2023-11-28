# pip install opencv-python mediapipe
import cv2
import mediapipe as mp

# Llamado al metodo face_detection de mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Indice de confianza --> Se eliminan todas las imagenes que tengan un valor menor al seleccionado
# A mayor indice de confianza, mayor seguridad de que sea una cara
with mp_face_detection.FaceDetection(
    min_detection_confidence=0.7) as face_detection:

    # Leer imagen
    image = cv2.imread("ImagenesPrueba\Foto.jpg")
    alto, ancho, _ = image.shape 
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Tranformar de bgr a rgb
    results = face_detection.process(image_rgb)

    # Se imprime el indice de confianza de la imagen y coordenadas de los 6 puntos clave
    # Ojo Izquierdo, Ojo derecho, Punta Nariz, Punta Boca, Oreja Izquierda, Oreja Derecha
    print("Resultados Imagen: ", results.detections)

    if results.detections is not None:
            for detection in results.detections:
                # Bounding box
                print(int(detection.location_data.relative_bounding_box.xmin * ancho))
                xmin = int(detection.location_data.relative_bounding_box.xmin * ancho)
                ymin = int(detection.location_data.relative_bounding_box.ymin * alto)
                w = int(detection.location_data.relative_bounding_box.width * ancho)
                h = int(detection.location_data.relative_bounding_box.height * alto)
                cv2.rectangle(image, (xmin, ymin), (xmin + w, ymin + h), (0, 255, 0), 2)
            
                # Ojo derecho - Right Eye (RE)
                x_RE = int(detection.location_data.relative_keypoints[0].x * ancho)
                y_RE = int(detection.location_data.relative_keypoints[0].y * alto)
                cv2.circle(image, (x_RE, y_RE), 3, (0, 0, 255), 2)

                # Ojo izquierdo - Left Eye (LE)
                x_LE = int(detection.location_data.relative_keypoints[1].x * ancho)
                y_LE = int(detection.location_data.relative_keypoints[1].y * alto)
                cv2.circle(image, (x_LE, y_LE), 3, (255, 0, 255), 2)

                # Punta de la Nariz - Nose Tip (NT)
                x_NT = int(detection.location_data.relative_keypoints[2].x * ancho)
                y_NT = int(detection.location_data.relative_keypoints[2].y * alto)
                cv2.circle(image, (x_NT, y_NT), 3, (255, 0, 255), 2)

                 # Centro de la boca - Mouth Center (MC)
                x_MC = int(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.MOUTH_CENTER).x * ancho)
                y_MC = int(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.MOUTH_CENTER).y * alto)
                cv2.circle(image, (x_MC, y_MC), 3, (0, 255, 0), 2)
                
                # Trago de la oreja derecha - Right Ear Tragion (RET)
                x_RET = int(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION).x * ancho)
                y_RET = int(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION).y * alto)
                cv2.circle(image, (x_RET, y_RET), 3, (0, 255, 255), 2)

                
                # Trago de la oreja izquierda - Left Ear Tragion (LET)
                x_LET = int(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION).x * ancho)
                y_LET = int(mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION).y * alto)
                cv2.circle(image, (x_LET, y_LET), 3, (255, 255, 0), 2)

    cv2.imshow("image", image)
    cv2.waitKey(0)
cv2.destroyAllWindows