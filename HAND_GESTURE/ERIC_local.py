"""
ERIC - Local Version
Hand gesture recognition using Hugging Face model, adapted for local machine use
"""
# Importar librerías necesarias
from flask import Flask, render_template, request, jsonify
from PIL import Image
import base64
import io
from transformers import pipeline
import os
import warnings

# Suprimir advertencias de PyTorch/HuggingFace para una salida más limpia
warnings.filterwarnings("ignore")

# Configuración de Flask
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Crear carpeta de templates si no existe
os.makedirs('templates', exist_ok=True)

# Cargar el modelo solo una vez
print("🔄 Cargando el modelo de reconocimiento de gestos...")
pipe = pipeline("image-classification", model="prithivMLmods/Hand-Gesture-19")
print("✅ Modelo cargado correctamente!")

@app.route('/')
def index():
    """Renderiza la página principal con la cámara web"""
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    """Procesa la imagen capturada desde la cámara web"""
    try:
        # Obtener los datos de la imagen
        data = request.json
        data_url = data['image']
        
        # Decodificar la imagen
        header, encoded = data_url.split(",", 1)
        img_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(img_data))
        
        # Clasificar con Hugging Face
        result = pipe(image)
        
        # Devolver resultado
        return jsonify({
            'success': True, 
            'result': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Crear archivo template HTML
def create_template():
    """Crea el archivo HTML para la interfaz web"""
    html_content = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ERIC - Reconocimiento de Gestos</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            text-align: center;
        }
        h1 {
            color: #333;
        }
        video, #captured-image {
            width: 400px;
            height: 300px;
            border: 1px solid #ccc;
            border-radius: 5px;
            margin: 10px auto;
            background-color: #f8f8f8;
        }
        button {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 10px 20px;
            text-align: center;
            font-size: 16px;
            margin: 10px 2px;
            cursor: pointer;
            border-radius: 5px;
        }
        button:hover {
            background-color: #45a049;
        }
        #result {
            margin-top: 20px;
            padding: 10px;
            border-radius: 5px;
            background-color: #f1f1f1;
        }
        .result-item {
            margin: 5px 0;
            padding: 5px;
            border-left: 3px solid #4CAF50;
            background-color: #f9f9f9;
            text-align: left;
        }
    </style>
</head>
<body>
    <h1>ERIC - Reconocimiento de Gestos</h1>
    <p>Versión local para reconocimiento de gestos de mano</p>
    
    <div>
        <video id="video" autoplay playsinline></video>
    </div>
    
    <div>
        <button id="capture-btn">📸 Capturar Imagen</button>
    </div>
    
    <div>
        <canvas id="canvas" style="display:none;"></canvas>
        <img id="captured-image" style="display:none;">
    </div>
    
    <div id="result">
        <p>Captura una imagen para obtener resultados</p>
    </div>

    <script>
        // Variables
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const captureButton = document.getElementById('capture-btn');
        const capturedImage = document.getElementById('captured-image');
        const resultDiv = document.getElementById('result');
        
        // Iniciar la cámara
        async function startCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
            } catch (err) {
                console.error("Error al acceder a la cámara: ", err);
                alert("No se pudo acceder a la cámara. Verifica los permisos.");
            }
        }
        
        // Capturar imagen
        captureButton.addEventListener('click', function() {
            // Configurar canvas
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            // Dibujar imagen en canvas
            const context = canvas.getContext('2d');
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // Convertir a imagen
            const dataUrl = canvas.toDataURL('image/jpeg');
            capturedImage.src = dataUrl;
            capturedImage.style.display = 'block';
            video.style.display = 'none';
            
            // Mostrar mensaje de procesamiento
            resultDiv.innerHTML = '<p>Procesando imagen...</p>';
            
            // Enviar al servidor
            processImage(dataUrl);
        });
        
        // Procesar imagen en servidor
        async function processImage(dataUrl) {
            try {
                const response = await fetch('/process_image', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ image: dataUrl })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    // Mostrar resultados
                    let resultsHtml = '<h3>Resultados:</h3>';
                    
                    data.result.forEach((item, index) => {
                        const confidence = (item.score * 100).toFixed(2);
                        resultsHtml += `<div class="result-item">
                            <strong>${item.label}</strong>: ${confidence}% de confianza
                        </div>`;
                    });
                    
                    resultsHtml += '<button id="restart-btn">🔄 Nueva captura</button>';
                    resultDiv.innerHTML = resultsHtml;
                    
                    // Añadir evento para volver a mostrar la cámara
                    document.getElementById('restart-btn').addEventListener('click', function() {
                        video.style.display = 'block';
                        capturedImage.style.display = 'none';
                        resultDiv.innerHTML = '<p>Captura una imagen para obtener resultados</p>';
                    });
                } else {
                    resultDiv.innerHTML = `<p>Error: ${data.error}</p>
                        <button id="restart-btn">🔄 Intentar de nuevo</button>`;
                        
                    document.getElementById('restart-btn').addEventListener('click', function() {
                        video.style.display = 'block';
                        capturedImage.style.display = 'none';
                        resultDiv.innerHTML = '<p>Captura una imagen para obtener resultados</p>';
                    });
                }
            } catch (err) {
                console.error("Error al procesar la imagen: ", err);
                resultDiv.innerHTML = `<p>Error de conexión al servidor</p>
                    <button id="restart-btn">🔄 Intentar de nuevo</button>`;
                    
                document.getElementById('restart-btn').addEventListener('click', function() {
                    video.style.display = 'block';
                    capturedImage.style.display = 'none';
                    resultDiv.innerHTML = '<p>Captura una imagen para obtener resultados</p>';
                });
            }
        }
        
        // Iniciar la aplicación
        startCamera();
    </script>
</body>
</html>"""
    
    # Guardar el template
    with open('templates/index.html', 'w') as f:
        f.write(html_content)
    print("✅ Template HTML creado correctamente")

def main():
    """Función principal para iniciar la aplicación"""
    # Crear el template HTML
    create_template()
    
    # Iniciar la aplicación Flask
    print("🚀 Iniciando servidor de ERIC...")
    print("🌐 Abre tu navegador en http://127.0.0.1:5000")
    app.run(debug=True)

if __name__ == '__main__':
    main()
