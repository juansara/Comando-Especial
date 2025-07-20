import requests
import time
import cv2
import numpy as np

# ------------ CONFIGURACIÓN ------------
ENDPOINT_URL = "http://127.0.0.1:5000/infer"

# Puedes reemplazar esta ruta por otra imagen válida
IMAGE_PATH = "test.jpg"  # Asegúrate de que este archivo exista

# Opcionales
REMOVE_BG = True
USE_AUGMENTATION = False
IMAGE_ONLY = False

# ------------ FUNCIÓN DE ENVÍO Y MEDICIÓN ------------
def medir_tiempo_inferencia():
    # Cargar imagen desde archivo y convertirla a JPG
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"❌ No se pudo leer la imagen en {IMAGE_PATH}")
        return

    _, buffer = cv2.imencode('.jpg', img)
    img_bytes = buffer.tobytes()

    files = {
        "image": ("image.jpg", img_bytes, "image/jpeg")
    }

    data = {
        "remove_bg": str(REMOVE_BG).lower(),
        "use_augmentation": str(USE_AUGMENTATION).lower(),
        "image_only": str(IMAGE_ONLY).lower()
    }

    # Medir tiempo
    t0 = time.perf_counter()
    try:
        response = requests.post(ENDPOINT_URL, files=files, data=data, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Error durante la petición: {e}")
        return
    t1 = time.perf_counter()

    rtt = (t1 - t0) * 1000
    print(f"✅ Tiempo de respuesta: {rtt:.1f} ms")
    print("➡️ Respuesta:", response.json())


# ------------ EJECUCIÓN ------------
if __name__ == "__main__":
    medir_tiempo_inferencia()
