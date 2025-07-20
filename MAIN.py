import cv2
import requests
import base64
import json
import time
import threading
import time
from typing import Optional, List, Tuple
import pydirectinput
import pyautogui

# ---------------- Configuración ----------------
CAM_INDEX = 0
FPS_VIDEO = 60  # Leer frames a esta tasa para fluidez (aunque no se muestren)
PERIODO_INFERENCIA_MS = 10  # Inferencia cada X milisegundos
TECLA_HOLD_TIME = 10  # Tiempo que se mantiene la tecla presionada (en segundos)

# Shared vector state for mouse movement
current_vector = [0, 0]
vector_lock = threading.Lock()

class MouseMover(threading.Thread):
    """Thread that continuously moves the mouse based on the current vector."""
    def __init__(self, fps: int = 30):
        """
        Initialize the mouse mover thread.
        
        Args:
            fps: Frames per second for mouse movement updates
        """
        super().__init__(daemon=True)
        self.running = True
        self.fps = fps
        self.frame_time = 1.0 / fps
        self.speed = 1.0  # Adjust this to control movement speed
        
    def run(self):
        """Main loop for continuous mouse movement."""
        while self.running:
            start_time = time.time()
            
            with vector_lock:
                # Get the current vector and apply speed
                dx, dy = current_vector[0] * self.speed, current_vector[1] * self.speed
            
            # Only move if there's actual movement
            if abs(dx) >= 1.0 or abs(dy) >= 1.0:
                # Convert to integers as pydirectinput requires them
                dx_int = int(round(dx))
                dy_int = int(round(dy))
                
                if dx_int != 0 or dy_int != 0:
                    try:
                        pydirectinput.moveRel(dx_int, -dy_int, relative=True, _pause=False)
                    except Exception as e:
                        print(f"Mouse movement error: {e}")
            
            # Calculate sleep time to maintain desired FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, self.frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
    def stop(self):
        """Stop the mouse movement thread."""
        self.running = False

class CameraStreamer:
    def __init__(self, endpoint_url: str, camera_index: int = 0, fps: int = 10):
        """
        Initialize the camera streamer.
        
        Args:
            endpoint_url: The URL endpoint to send frames to
            camera_index: Camera index (0 for default camera)
            fps: Frames per second to capture and send
        """
        self.endpoint_url = endpoint_url
        self.camera_index = camera_index
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self.cap = None
        self.running = False
        self.capture_thread = None
        self.mouse_mover = None
        
    def initialize_camera(self) -> bool:
        """Initialize the camera capture."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_index}")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            print(f"Camera initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def encode_frame(self, frame) -> bytes:
        """Encode frame to JPEG bytes."""
        try:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return buffer.tobytes()
        except Exception as e:
            print(f"Error encoding frame: {e}")
            return None
    
    def send_frame(self, frame_data: bytes) -> bool:
        """Send frame data to the endpoint as multipart form data."""
        try:
            # Create multipart form data
            files = {
                'image': ('frame.jpg', frame_data, 'image/jpeg')
            }
            
            # Optional: Add additional form fields
            data = {
                'timestamp': str(time.time()),
                'format': 'jpeg'
            }
            
            response = requests.post(
                self.endpoint_url, 
                files=files,
                data=data,
                timeout=5
            )
            
            
            if response.status_code == 200:

                data = response.json()
                print(data)
                orientation = data['orientation']
                gesture = data['gesture']
                label = gesture['label']
                center_x = orientation[0][0]
                center_y = orientation[0][1]
                end_x = orientation[1][0]
                end_y = orientation[1][1]

                # Calculate direction vector from center to gaze point
                move_x = end_x - center_x
                move_y = end_y - center_y
                
                # Normalize the vector
                length = (move_x**2 + move_y**2) ** 0.5
                if length > 0:
                    move_x = move_x / length
                    move_y = move_y / length
                
                # Update the shared vector
                with vector_lock:
                    current_vector[0] = move_x * 5  # Scale factor for speed
                    current_vector[1] = move_y * 5

                if label == 'one':
                    pyautogui.keyDown('w')
                    time.sleep(TECLA_HOLD_TIME)
                    pyautogui.keyUp('w')
                elif label == 'peace':
                    pyautogui.keyDown('a')
                    time.sleep(TECLA_HOLD_TIME)
                    pyautogui.keyUp('a')
                elif label == 'palm':
                    pyautogui.keyDown('s')
                    time.sleep(TECLA_HOLD_TIME)
                    pyautogui.keyUp('s')
                elif label == 'like':
                    pyautogui.keyDown('d')
                    time.sleep(TECLA_HOLD_TIME)
                    pyautogui.keyUp('d')
                elif label == 'fist':
                    pyautogui.click(button='left')
                elif label == 'ok':
                    pyautogui.click(button='right')
                elif label == 'rock':
                    pyautogui.press('space')
            #     gaze_vector = data['gaze_vector']
            #     end_x = gaze_vector['end_x']
            #     end_y = gaze_vector['end_y']
            #     start_x = gaze_vector['start_x']
            #     start_y = gaze_vector['start_y']

            #     x_move = end_x - start_x
            #     y_move = end_y - start_y

            #     mouse.move(x_move, y_move)

            #     print(data)
            #     return True
            # else:
            #     print(f"Server responded with status code: {response.status_code}")
            #     return False
                
        except requests.exceptions.RequestException as e:
            print(f"Error sending frame: {e}")
            return False
    
    def capture_and_send_loop(self):
        """Main loop for capturing and sending frames."""
        last_frame_time = 0
        
        while self.running:
            current_time = time.time()
            
            # Control frame rate
            if current_time - last_frame_time < self.frame_interval:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                continue
            
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Encode frame
            frame_data = self.encode_frame(frame)
            if frame_data:
                # Send frame to endpoint
                success = self.send_frame(frame_data)
                if success:
                    print(f"Frame sent successfully at {time.strftime('%H:%M:%S')}")
                else:
                    print("Failed to send frame")
            
            last_frame_time = current_time
            
            # Optional: Display the frame locally (comment out if not needed)
            # cv2.imshow('Camera Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break
    
    def start(self):
        """Start the camera streaming and mouse movement."""
        if not self.initialize_camera():
            return False
        
        # Start mouse movement thread
        self.mouse_mover = MouseMover()
        self.mouse_mover.start()
        
        # Start camera capture thread
        self.running = True
        self.capture_thread = threading.Thread(target=self.capture_and_send_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        print(f"Camera streaming started. Sending frames to {self.endpoint_url}")
        print("Press 'q' in the camera window to stop, or call stop() method")
        return True
    
    def stop(self):
        """Stop the camera streaming and mouse movement."""
        self.running = False
        
        # Stop mouse movement
        if self.mouse_mover:
            self.mouse_mover.stop()
            self.mouse_mover.join(timeout=1)
        
        # Stop camera capture
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        
        # Release camera resources
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("Camera streaming and mouse movement stopped")


def main():
    # Configuration
    ENDPOINT_URL = "http://127.0.0.1:5000/infer"  # Change this to your endpoint
    CAMERA_INDEX = 0  # 0 for default camera, 1, 2, etc. for other cameras
    FPS = 10  # Frames per second
    
    # Create and start the camera streamer
    streamer = CameraStreamer(ENDPOINT_URL, CAMERA_INDEX, FPS)
    
    try:
        if streamer.start():
            # Keep the main thread alive
            while streamer.running:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    finally:
        streamer.stop()


if __name__ == "__main__":
    main()
