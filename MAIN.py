import cv2
import requests
import time
import threading
from typing import Optional
import pydirectinput
import pyautogui

# ---------------- Configuración ----------------
CAM_INDEX       = 0
FPS_VIDEO       = 60          # FPS de lectura de cámara
FPS_ENVÍO       = 10          # FPS al servidor
MOUSE_SPEED     = 5           # Escala de movimiento del ratón

TECLAS_SOSTENIDAS = {         # Gestos que mantienen tecla
    'one':  'w',
    'peace':'a',
    'palm': 's',
    'like': 'd',
}

CLICK_GESTURES = {            # Gestos de golpe/clic
    'fist': ('left',),            # clic izquierdo
    'ok':   ('right',),           # clic derecho
    'rock': ('press', 'space'),   # barra espaciadora
}
CLICK_INTERVAL = 0.01       # segundos entre clics repetidos

# ---------------- Estado compartido ----------------
current_vector = [0.0, 0.0]
vector_lock    = threading.Lock()


class MouseMover(threading.Thread):
    """Hilo que mueve el ratón en función de current_vector."""
    def __init__(self, fps:int = 30):
        super().__init__(daemon=True)
        self.running    = True
        self.frame_time = 1.0 / fps
        self.speed      = 1.0

    def run(self):
        while self.running:
            t0 = time.time()
            with vector_lock:
                dx = current_vector[0] * self.speed
                dy = current_vector[1] * self.speed
            if abs(dx) >= 1 or abs(dy) >= 1:
                try:
                    pydirectinput.moveRel(int(round(dx)),
                                          -int(round(dy)),
                                          relative=True,
                                          _pause=False)
                except Exception as e:
                    print(f"Mouse movement error: {e}")
            time.sleep(max(0, self.frame_time - (time.time() - t0)))

    def stop(self):
        self.running = False


class CameraStreamer:
    def __init__(self,
                 endpoint_url:str,
                 camera_index:int = 0,
                 fps:int = FPS_ENVÍO):
        self.endpoint_url  = endpoint_url
        self.camera_index  = camera_index
        self.fps           = fps
        self.frame_interval = 1.0 / fps

        self.cap           = None
        self.running       = False
        self.capture_thread= None
        self.mouse_mover   = None

        self.active_key: Optional[str] = None   # tecla mantenida
        self.last_click_time           = 0.0    # control de ráfaga para clics

    # ---------- Cámara ----------
    def initialize_camera(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"[ERROR] No se pudo abrir la cámara {self.camera_index}")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, FPS_VIDEO)
        print("[INFO] Cámara inicializada correctamente")
        return True

    @staticmethod
    def encode_frame(frame):
        resized = cv2.resize(frame, (320, 240))  # menor resolución
        ok, buf = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return buf.tobytes() if ok else None

    # ---------- Envío y manejo de respuesta ----------
    def send_frame(self, frame_data:bytes):
        files = {'image': ('frame.jpg', frame_data, 'image/jpeg')}
        data  = {'timestamp': str(time.time()), 'format': 'jpeg'}

        t0 = time.perf_counter()                    # ← añadido
        try:
            r = requests.post(self.endpoint_url, files=files, data=data, timeout=5)
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Error sending frame: {e}")
            return
        latency_ms = (time.perf_counter() - t0) * 1000   # ← añadido
        print(f"[RTT] {latency_ms:.1f} ms")              # ← añadido

        if r.status_code != 200:
            print(f"[WARN] Servidor respondió {r.status_code}")
            return

        payload = r.json()
        orientation = payload['orientation']
        gesture_lbl = payload['gesture']['label']
        print(f"[DEBUG] Gesto: {gesture_lbl} | Orientación: {orientation}")

        # ---------- MOVER RATÓN ----------
        (cx, cy), (ex, ey) = orientation
        mvx, mvy = ex - cx, ey - cy
        length = (mvx**2 + mvy**2) ** 0.5
        if length:
            mvx, mvy = (mvx/length)*MOUSE_SPEED, (mvy/length)*MOUSE_SPEED
            with vector_lock:
                current_vector[0], current_vector[1] = mvx, mvy

        # ---------- TECLAS SOSTENIDAS ----------
        new_key = TECLAS_SOSTENIDAS.get(gesture_lbl)

        if new_key != self.active_key:
            if self.active_key:
                print(f"[KEY] Soltando {self.active_key}")
                pyautogui.keyUp(self.active_key)
            if new_key:
                print(f"[KEY] Presionando {new_key}")
                pyautogui.keyDown(new_key)
            self.active_key = new_key

        # ---------- CLICS REPETIDOS ----------
        if gesture_lbl in CLICK_GESTURES:
            now = time.time()
            if now - self.last_click_time >= CLICK_INTERVAL:
                action = CLICK_GESTURES[gesture_lbl]
                if action[0] == 'left':
                    print("[CLICK] Left click")
                    pyautogui.click(button='left')
                elif action[0] == 'right':
                    print("[CLICK] Right click")
                    pyautogui.click(button='right')
                elif action[0] == 'press':
                    print(f"[CLICK] Press {action[1]}")
                    pyautogui.press(action[1])
                self.last_click_time = now
        else:
            # Si el gesto no es de clic, reinicia el temporizador
            self.last_click_time = 0.0

    # ---------- Bucle de captura ----------
    def capture_and_send_loop(self):
        last_time = 0.0
        while self.running:
            if time.time() - last_time < self.frame_interval:
                time.sleep(0.001)
                continue

            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] No se pudo leer frame")
                break

            data = self.encode_frame(frame)
            if data:
                self.send_frame(data)

            last_time = time.time()

            # (opcional) mostrar feed
            # cv2.imshow('Feed', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     self.stop(); break

    # ---------- Control ----------
    def start(self):
        if not self.initialize_camera():
            return False

        self.mouse_mover = MouseMover()
        self.mouse_mover.start()

        self.running = True
        self.capture_thread = threading.Thread(target=self.capture_and_send_loop,
                                               daemon=True)
        self.capture_thread.start()

        print(f"[INFO] Streaming to {self.endpoint_url}. Ctrl+C para salir.")
        return True

    def stop(self):
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(1)

        if self.mouse_mover:
            self.mouse_mover.stop()
            self.mouse_mover.join(1)

        if self.active_key:
            pyautogui.keyUp(self.active_key)
            print(f"[KEY] Soltando {self.active_key}")
            self.active_key = None

        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera streaming and mouse movement stopped")


def main():
    ENDPOINT = "http://127.0.0.1:5000/infer"
    streamer = CameraStreamer(ENDPOINT, camera_index=CAM_INDEX)

    try:
        if streamer.start():
            while streamer.running:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupción del usuario")
    finally:
        streamer.stop()

if __name__ == "__main__":
    main()
