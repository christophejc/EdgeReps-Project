from flask import Flask, Response
import cv2
import time
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Initialize model
yolo_model = YOLO("yolo26n-pose.pt") 

# COCO Connections for Drawing
CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (12, 14), (13, 15), (14, 16)
]

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    return angle

def generate_frames():
    cap = cv2.VideoCapture(0)
    # Set resolution for C270 / Pi 5 balance
    width, height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Rep Counter Variables
    counter = 0
    stage = None
    
    #Performance variables
    avg_fps = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        start_time = time.time()
        
        # Inference (Using imgsz=320 for speed optimization)
        results = yolo_model.predict(frame, imgsz=320, conf=0.5, verbose=False, device="cpu", stream=True)

        for r in results:
            if r.keypoints is not None and len(r.keypoints.xy) > 0:
                # 1. Get Coordinates & Confidences
                kpts = r.keypoints.xy[0].cpu().numpy()
                conf = r.keypoints.conf[0].cpu().numpy()

                try:
                    # 2. Extract specific joints for push-up logic
                    # 5:L_Shoulder, 7:L_Elbow, 9:L_Wrist | 6:R_Shoulder, 8:R_Elbow, 10:R_Wrist
                    l_sh, l_el, l_wr = kpts[5], kpts[7], kpts[9]
                    r_sh, r_el, r_wr = kpts[6], kpts[8], kpts[10]

                    # 3. Calculate Angles
                    angle_L = calculate_angle(l_sh, l_el, l_wr)
                    angle_R = calculate_angle(r_sh, r_el, r_wr)

                    # 4. Push-up Logic
                    # Up position: arms extended (> 160) | Down position: arms bent (< 90)
                    if angle_L > 160 and angle_R > 150:
                        if stage == "down":
                            counter += 1
                        stage = "up"
                    elif angle_L < 90 and angle_R < 90:
                        stage = "down"

                    # 5. Draw Angles near elbows
                    cv2.putText(frame, str(int(angle_L)), tuple(l_el.astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, str(int(angle_R)), tuple(r_el.astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                except Exception:
                    pass

                # 6. Draw Connections (Lines)
                for start, end in CONNECTIONS:
                    if conf[start] > 0.6 and conf[end] > 0.6:
                        pt1, pt2 = kpts[start], kpts[end]
                        cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 42, 4), 2)

                # 7. Draw Keypoints (Dots)
                for i, pt in enumerate(kpts):
                    if conf[i] > 0.6:
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (186, 0, 221), -1)

        # Performance and Dashboard UI
        process_time = time.time() - start_time
        fps = 1.0 / process_time if process_time > 0 else 0
        avg_fps.append(fps)
        if len(avg_fps) > 20: avg_fps.pop(0)
        
        # Dashboard Overlay
        cv2.rectangle(frame, (0,0), (220, 80), (104, 0, 123), -1)
        cv2.putText(frame, f'REPS: {counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f'STAGE: {stage}', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
        # Right-aligned FPS
        fps_text = f"FPS: {int(sum(avg_fps)/len(avg_fps))}"
        cv2.putText(frame, fps_text, (width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<html><body style='margin:0; background:black;'><img src='/video_feed' width='100%'></body></html>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)