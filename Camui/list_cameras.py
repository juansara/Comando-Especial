import cv2

def list_available_cameras(max_tests=10):
    """List all available cameras by testing indices 0 to max_tests-1"""
    available_cameras = []
    
    print("Searching for cameras...")
    for i in range(max_tests):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret = cap.read()[0]
            if ret:
                info = f"Camera {i}: "
                info += f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
                info += f"{cap.getBackendName()} backend"
                available_cameras.append(i)
                print(info)
            cap.release()
    
    if not available_cameras:
        print("No cameras found!")
    
    return available_cameras

if __name__ == "__main__":
    list_available_cameras()
