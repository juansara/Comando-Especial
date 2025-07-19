import tkinter as tk
from datetime import datetime
from pathlib import Path
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont

# ---------------- Configuración ----------------
CAM_INDEX   = 0               # índice de la webcam
FPS_VIDEO   = 30              # refresco de la vista previa
PERIODO_FOTO_MS = 1000        # 1000 ms  →  1 s
CARPETA_SALIDA = Path(__file__).parent

class AutoCamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Captura automática 1 foto/s")
        self.root.protocol("WM_DELETE_WINDOW", self.cerrar)

        self.lbl_video = tk.Label(root)
        self.lbl_video.pack()

        # abrir cámara
        self.cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
        self.cam_activa = self.cap.isOpened()
        if self.cam_activa:
            self.cap.set(cv2.CAP_PROP_FPS, FPS_VIDEO)
        else:
            print("⚠️  No se encontró cámara; se mostrará recuadro gris.")

        self.frame_actual = None
        self.refrescar()
        self.programar_foto()

    # ---------- Mostrar vídeo / recuadro gris ----------
    def refrescar(self):
        if self.cam_activa:
            ok, frame = self.cap.read()
            if ok:
                self.frame_actual = frame.copy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
            else:
                img = self.marco_gris()
        else:
            img = self.marco_gris()

        imgtk = ImageTk.PhotoImage(img)
        self.lbl_video.imgtk = imgtk
        self.lbl_video.configure(image=imgtk)

        self.root.after(int(1000 / FPS_VIDEO), self.refrescar)

    # ---------- Guardar foto cada segundo ----------
    def programar_foto(self):
        if self.cam_activa and self.frame_actual is not None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            nombre = CARPETA_SALIDA / f"foto_{ts}.jpg"
            cv2.imwrite(str(nombre), self.frame_actual)
            print(f"📸  Guardada {nombre.name}")

        # volver a programar dentro de 1 s
        self.root.after(PERIODO_FOTO_MS, self.programar_foto)

    # ---------- Marco gris con texto ----------
    def marco_gris(self):
        img = Image.new("RGB", (640, 480), "gray")
        draw = ImageDraw.Draw(img)
        msg = "Cámara no disponible"
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), msg, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((640 - w) / 2, (480 - h) / 2), msg, fill="white", font=font)
        return img

    def cerrar(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

# ---------------- Ejecutar ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app  = AutoCamApp(root)
    root.mainloop()