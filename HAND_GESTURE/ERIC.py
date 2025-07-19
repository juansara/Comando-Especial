# Paso 1: Importar librerías necesarias
from IPython.display import display, Javascript, HTML
from google.colab import output
from PIL import Image
import base64
import io

# Paso 2: Crear y cargar el modelo de Hugging Face
from transformers import pipeline

# Cargar el modelo solo una vez
pipe = pipeline("image-classification", model="prithivMLmods/Hand-Gesture-19")

# Paso 3: Mostrar cámara y botón de captura
def show_camera():
    display(HTML('''
        <video autoplay playsinline id="video" width="400" height="300"></video>
        <br>
        <button onclick="captureAndSend()">📸 Capturar Imagen</button>
        <script>
        const video = document.getElementById('video');
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
            })
            .catch(err => console.error("Error accessing camera: ", err));

        async function captureAndSend() {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const context = canvas.getContext('2d');
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const dataUrl = canvas.toDataURL('image/jpeg');
            google.colab.kernel.invokeFunction('notebook.process_image', [dataUrl], {});
        }
        </script>
    '''))

# Paso 4: Procesar imagen y usar modelo
def process_image(data_url):
    print("📥 Imagen recibida")

    # Decodificar la imagen
    header, encoded = data_url.split(",", 1)
    data = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(data))

    # Mostrar la imagen capturada
    display(image)

    # Clasificar con Hugging Face
    try:
        result = pipe(image)
        print("🔍 Resultado del modelo:", result)
    except Exception as e:
        print(f"❌ Error al clasificar imagen: {e}")

# Paso 5: Registrar el callback y mostrar cámara
output.register_callback('notebook.process_image', process_image)
show_camera()