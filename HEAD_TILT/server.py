import gradio as gr
from paths import *

from vision_tower import DINOv2_MLP
from transformers import AutoImageProcessor
import torch
from inference import *
from utils import *

from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download(repo_id="Viglong/Orient-Anything", filename="croplargeEX2/dino_weight.pt", repo_type="model", cache_dir='./', resume_download=True)
print(ckpt_path)

save_path = './'
device = 'cpu'
dino = DINOv2_MLP(
                    dino_mode   = 'large',
                    in_dim      = 1024,
                    out_dim     = 360+180+180+2,
                    evaluate    = True,
                    mask_dino   = False,
                    frozen_back = False
                )

dino.eval()
print('model create')
dino.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
dino = dino.to(device)
print('weight loaded')
val_preprocess   = AutoImageProcessor.from_pretrained(DINO_LARGE, cache_dir='./')

def infer_func(img, do_rm_bkg, do_infer_aug):
    origin_img = Image.fromarray(img)
    if do_infer_aug:
        rm_bkg_img = background_preprocess(origin_img, True)
        angles = get_3angle_infer_aug(origin_img, rm_bkg_img, dino, val_preprocess, device)
    else:
        rm_bkg_img = background_preprocess(origin_img, do_rm_bkg)
        angles = get_3angle(rm_bkg_img, dino, val_preprocess, device)
    
    phi   = np.radians(angles[0])
    theta = np.radians(angles[1])
    gamma = angles[2]
    confidence = float(angles[3])
    if confidence > 0.5:
        render_axis = render_3D_axis(phi, theta, gamma)
        res_img = overlay_images_with_scaling(render_axis, rm_bkg_img)
    else:
        res_img = img
    
    # axis_model = "axis.obj"
    return [res_img, round(float(angles[0]), 2), round(float(angles[1]), 2), round(float(angles[2]), 2), round(float(angles[3]), 2)]

from flask import Flask, request, jsonify, Response
import numpy as np
import cv2
import io
from PIL import Image
import json

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/infer', methods=['POST'])
def process_image():
    """
    Process an image buffer and return inference results
    
    Expected POST data:
    - image: binary image data (required)
    - remove_bg: boolean (optional, default=True)
    - use_augmentation: boolean (optional, default=False)
    """
    try:
        # Check if the post request has the image part
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
            
        # Read image from request
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        # Read image into numpy array
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get optional parameters
        remove_bg = request.form.get('remove_bg', 'true').lower() == 'true'
        use_augmentation = request.form.get('use_augmentation', 'false').lower() == 'true'
        image_only = request.form.get('image_only', 'false').lower() == 'true'
        
        # Perform inference
        result = infer_func(img, remove_bg, use_augmentation)
        
        # Convert the result image back to BGR for OpenCV drawing
        output_img = cv2.cvtColor(np.array(result[0]), cv2.COLOR_RGB2BGR)
        height, width = output_img.shape[:2]
        
        # Calculate gaze vector endpoints (simplified 2D projection)
        azimuth_rad = np.radians(float(result[1]))
        polar_rad = np.radians(float(result[2]))
        
        # Calculate 2D vector components (simplified projection)
        length = min(height, width) * 0.4  # Vector length as fraction of image size
        end_x = int(width/2 + np.cos(azimuth_rad) * length)
        end_y = int(height/2 - np.sin(azimuth_rad) * length * np.cos(polar_rad))
        
        # Define center point
        center = (width // 2, height // 2)
        
        # Line 1: Main gaze direction (green)
        gaze_end = (end_x, end_y)
        cv2.arrowedLine(output_img, center, gaze_end, (0, 255, 0), 2, tipLength=0.2)
        
        # Line 2: Horizontal rotation (blue)
        horiz_length = min(height, width) * 0.3
        horiz_end = (
            int(center[0] + np.cos(azimuth_rad) * horiz_length),
            center[1]  # Keep same y-coordinate for horizontal movement
        )
        cv2.line(output_img, center, horiz_end, (255, 0, 0), 2)
        
        # Line 3: Vertical tilt (red)
        vert_length = min(height, width) * 0.3
        vert_end = (
            center[0],  # Keep same x-coordinate for vertical movement
            int(center[1] - np.sin(polar_rad) * vert_length)
        )
        cv2.line(output_img, center, vert_end, (0, 0, 255), 2)
        
        # Add a small circle at the center point (white)
        cv2.circle(output_img, center, 5, (255, 255, 255), -1)
        
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        cv2.putText(output_img, 'Gaze', (gaze_end[0] + 5, gaze_end[1]), 
                   font, font_scale, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(output_img, 'Yaw', (horiz_end[0] + 5, horiz_end[1]), 
                   font, font_scale, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(output_img, 'Pitch', (vert_end[0] + 5, vert_end[1]), 
                   font, font_scale, (0, 0, 255), 1, cv2.LINE_AA)
        
        # Convert the image to bytes
        _, img_encoded = cv2.imencode('.jpg', output_img)

        # Return just the image if requested
        if image_only:
            img_bytes = img_encoded.tobytes()
            return Response(
                img_bytes,
                mimetype='image/jpeg',
                headers={
                    'Content-Disposition': 'inline; filename=output.jpg',
                    'Content-Length': len(img_bytes)
                }
            )
        
        # Otherwise return JSON with image data and vector info
        response = {
            # 'image': img_encoded.tobytes().hex(),
            'gaze_vector': {
                'start_x': center[0],
                'start_y': center[1],
                'end_x': end_x,
                'end_y': end_y,
                'azimuth': float(result[1]),
                'polar': float(result[2]),
                'confidence': float(result[4])
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
