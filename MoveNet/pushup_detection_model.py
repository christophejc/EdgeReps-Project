from flask import Flask, Response
import cv2
import time
import tensorflow as tf
import numpy as np
import model as m  # Uses your existing model.py functions

app = Flask(__name__)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def generate_frames():
    # 1. Setup Camera and Dimensions
    cap = cv2.VideoCapture(0)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize the cropping region
    crop_region = m.init_crop_region(height, width)

    # Push-up Variables
    counter = 0
    stage = None
    
    #Performance variables
    avg_fps = []

    while True:
        success, frame = cap.read()
        
        if not success:
            break

        start_time = time.time()
        # 2. AI Inference
        # MoveNet expects RGB, OpenCV gives BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # EXACT CALL: matching your specific run_inference function
        keypoints_with_scores = m.run_inference_video(
            m.movenet, rgb_frame, crop_region, 
            crop_size=[m.input_size, m.input_size]
        )

        # 1. Coordinate Extraction using your KEYPOINT_DICT indices
        kpts = keypoints_with_scores[0, 0]

        try:
            # Map normalized MoveNet [y, x, s] to [x, y] for angle math
            l_shoulder = [kpts[m.KEYPOINT_DICT['left_shoulder']][1], kpts[m.KEYPOINT_DICT['left_shoulder']][0]]
            l_elbow    = [kpts[m.KEYPOINT_DICT['left_elbow']][1], kpts[m.KEYPOINT_DICT['left_elbow']][0]]
            l_wrist    = [kpts[m.KEYPOINT_DICT['left_wrist']][1], kpts[m.KEYPOINT_DICT['left_wrist']][0]]
            
            r_shoulder = [kpts[m.KEYPOINT_DICT['right_shoulder']][1], kpts[m.KEYPOINT_DICT['right_shoulder']][0]]
            r_elbow    = [kpts[m.KEYPOINT_DICT['right_elbow']][1], kpts[m.KEYPOINT_DICT['right_elbow']][0]]
            r_wrist    = [kpts[m.KEYPOINT_DICT['right_wrist']][1], kpts[m.KEYPOINT_DICT['right_wrist']][0]]

            # 2. Angle Calculation
            angle_L = calculate_angle(l_shoulder, l_elbow, l_wrist)
            angle_R = calculate_angle(r_shoulder, r_elbow, r_wrist)

            # 3. Logic: Check both arms to ensure proper form
            if angle_L > 160 and angle_R > 160:
                if stage == "down":
                    counter += 1
                stage = "up"
            elif angle_L < 80 and angle_R < 80:
                stage = "down"
            
            # 4. Draw Stats Dashboard
            cv2.rectangle(frame, (0,0), (250, 80), (245, 117, 16), -1)
            cv2.putText(frame, f'REPS: {counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(frame, f'STAGE: {stage}', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        except:
            pass

        # 1. DRAW SKELETON (Lines)
        for edge, color in m.KEYPOINT_EDGE_INDS_TO_COLOR_BGR.items():
            p1_idx, p2_idx = edge
            y1, x1, s1 = kpts[p1_idx]
            y2, x2, s2 = kpts[p2_idx]

            # Only draw line if both points are detected with high confidence
            if s1 > 0.4 and s2 > 0.4:
                pt1 = (int(x1 * width), int(y1 * height))
                pt2 = (int(x2 * width), int(y2 * height))
                cv2.line(frame, pt1, pt2, color, 2)

        # 2. DRAW KEYPOINTS (Dots)
        for i in range(17):
            y, x, s = kpts[i]
            if s > 0.5: # Confidence is at .4
                cv2.circle(frame, (int(x * width), int(y * height)), 5, (147, 20, 255), -1)

        # 3. DRAW ANGLES (Text)
        try:
            # We already calculated angle_L and angle_R in the logic above
            # Placing text near the elbows (indices 7 and 8)
            l_elbow_pos = (int(kpts[7][1] * width), int(kpts[7][0] * height))
            r_elbow_pos = (int(kpts[8][1] * width), int(kpts[8][0] * height))

            cv2.putText(frame, f"{int(angle_L)}deg", (l_elbow_pos[0] + 10, l_elbow_pos[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"{int(angle_R)}deg", (r_elbow_pos[0] + 10, r_elbow_pos[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        except:
            pass

        # Performance and Dashboard UI
        process_time = time.time() - start_time
        fps = 1.0 / process_time if process_time > 0 else 0
        avg_fps.append(fps)
        if len(avg_fps) > 20: avg_fps.pop(0)

        # Right-aligned FPS
        fps_text = f"FPS: {int(sum(avg_fps)/len(avg_fps))}"
        cv2.putText(frame, fps_text, (width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


        # 4. Update Crop & Finalize Frame
        crop_region = m.determine_crop_region(keypoints_with_scores, height, width)
        
        # Encode as JPG for the browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    # Direct link to the stream
    return "<html><body style='margin:0;'><img src='/video_feed' width='100%'></body></html>"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Access via http://192.168.100.1:5000
    app.run(host='0.0.0.0', port=5000, threaded=True)