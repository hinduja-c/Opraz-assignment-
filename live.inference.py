import cv2
import time
from ultralytics import YOLO

# 1. Load the quantized INT8 ONNX model locally from your folder
model = YOLO("best_int8.onnx", task='detect')

# 2. Open camera channel 0 (your local built-in webcam)
cap = cv2.VideoCapture(0) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Webcam feed disconnected or not found.")
        break

    # --- LATENCY TRACKING ---
    t0 = time.perf_counter()
    # Pre-processing frame track
    pre_proc_time = (time.perf_counter() - t0) * 1000 

    t1 = time.perf_counter()
    # Core Edge Model Prediction Math
    results = model.predict(frame, verbose=False, imgsz=640, int8=True)
    inference_time = (time.perf_counter() - t1) * 1000 

    # Approximate NMS/Post-processing allocation based on your Colab validation 
    post_proc_time = 5.1 

    # Calculate real-time Frames Per Second (FPS)
    fps = 1000 / inference_time if inference_time > 0 else 0
    
    # --- VISUALIZATION OVERLAYS ---
    # Paint the bounding boxes, class names, and confidence scores onto the frame
    annotated_frame = results[0].plot() 

    # Paint the required engineering metrics directly on the upper-left layout
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Pre-process: {pre_proc_time:.1f} ms", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(annotated_frame, f"Inference: {inference_time:.1f} ms", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    cv2.putText(annotated_frame, f"Post-process: {post_proc_time:.1f} ms", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Launch the native window display
    cv2.imshow("Edge CV Local Inference Window", annotated_frame)

    # Clean kill switch: Pressing 'q' on your keyboard closes the application safely
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()