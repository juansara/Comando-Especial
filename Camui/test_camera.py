import cv2

print("Testing default camera (index 0)...")
cap = cv2.VideoCapture(0)

if cap.isOpened():
    print("Success! Camera is working.")
    ret, frame = cap.read()
    if ret:
        print(f"Frame dimensions: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print("Warning: Could not read frame from camera")
    cap.release()
else:
    print("Error: Could not open camera")
    print("\nTroubleshooting steps:")
    print("1. Make sure your camera is properly connected")
    print("2. Check if another application is using the camera")
    print("3. On Linux, try installing v4l-utils:")
    print("   sudo apt update && sudo apt install v4l-utils")
    print("4. Check camera permissions")
    print("5. Try a different camera index (e.g., 1, 2, etc.)")
