from flask import Flask, Response
import cv2
import time
import mediapipe as mp
import numpy as np
import paho.mqtt.client as mqtt
import pyttsx3
import json
import threading
import os
from pi5neo import Pi5Neo

# MediaPipe Setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
app = Flask(__name__)

# --- GLOBAL STATE VARIABLES ---
is_running = False
current_exercise = "PUSHUP"
last_active_time = time.time()
counter = 0
stage = None
JOINT_MAP = {
    "PUSHUP": {
        "left": (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        "right": (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)
    },
    "SQUAT": {
        "left": (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
        "right": (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
    }
}


# --- VOICE ENGINE (Thread-Safe Wrapper) ---
engine = pyttsx3.init()
engine.setProperty('rate', 160)

def speak(text):
    # Running in a thread prevents the video feed from freezing while talking
    def target():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=target).start()

def check_audio():
    # Check if the Jieli chip is seen by the system
    if os.path.exists('/proc/asound/UACDemoV10'):
        print("✅ Audio Hardware: FOUND")
    else:
        print("❌ Audio Hardware: MISSING (Check USB connection)")

check_audio()

# ---LED CONFIGURATION ---
NUM_LEDS = 20
SPI_BUS = '/dev/spidev0.0'

# Initialize the strip globally
# This opens the SPI connection once when the script starts
neo = Pi5Neo(SPI_BUS, NUM_LEDS)

def flash_leds(color_type):
    def target():
        # Define colors based on success or error
        if color_type == "success":
            r, g, b = 0, 255, 0  # Green
        elif color_type == "error":
            r, g, b = 255, 0, 0  # Red
        else:
            r, g, b = 0, 0, 0    # Off

        # Execute the flash
        neo.fill_strip(r, g, b)
        neo.update_strip()
        time.sleep(0.3)  # How long the feedback stays visible
        neo.clear_strip()
        neo.update_strip()

    # Spawning the thread so MediaPipe keeps running
    threading.Thread(target=target).start()

# --- MQTT SETUP ---
def on_message(client, userdata, msg):
    global is_running, current_exercise, last_active_time, counter, stage
    try:
        data = json.loads(msg.payload.decode())
        if data.get("action") == "START" and not is_running:
            current_exercise = data.get("exercise", "PUSHUP") #Default parameter assignment
            print("Received MQTT!")
            is_running = True
            last_active_time = time.time()
            counter = 0
            stage = None
            speak(f"Starting {current_exercise} mode.")
        elif data.get("action") == "STOP":
            is_running = False
            speak("Session stopped.")
    except Exception as e:
        print(f"MQTT Error: {e}")

mqtt_client = mqtt.Client()
mqtt_client.on_message = on_message
try:
    mqtt_client.connect("localhost", 1883, 60)
    mqtt_client.subscribe("workout/control")
    mqtt_client.loop_start()
except Exception as e:
    print(f"Could not connect to MQTT: {e}")

# --- EXISTING UTILITY ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360-angle if angle > 180.0 else angle

def generate_frames():
    global is_running, last_active_time, counter, stage
    
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    avg_fps = []
    
    # Flags for intermediate/incomplete reps
    warned_depth = False 
    warned_height = False

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # 1. IDLE CHECK (Save CPU when not triggered by MQTT)
            if not is_running:
                cv2.putText(frame, "WAITING FOR MQTT...", (width//4, height//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.1)
                continue

            start_time = time.time()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 1. Logic for the Watchdog Timer
            time_since_detection = time.time() - last_active_time

            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    exercise_joints = JOINT_MAP[current_exercise]

                    # Calculate Left Side
                    L_A, L_B, L_C = exercise_joints["left"]
                    angle_L = calculate_angle(
                        [landmarks[L_A.value].x, landmarks[L_A.value].y],
                        [landmarks[L_B.value].x, landmarks[L_B.value].y],
                        [landmarks[L_C.value].x, landmarks[L_C.value].y]
                    )

                    # Calculate Right Side
                    R_A, R_B, R_C = exercise_joints["right"]
                    angle_R = calculate_angle(
                        [landmarks[R_A.value].x, landmarks[R_A.value].y],
                        [landmarks[R_B.value].x, landmarks[R_B.value].y],
                        [landmarks[R_C.value].x, landmarks[R_C.value].y]
                    )
                    # --- BILATERAL REP LOGIC ---
                    # Both arms/legs must be extended to be "UP"
                    if angle_L > 160 and angle_R > 160:
                        if stage == "down":
                            counter += 1
                            speak(str(counter))
                            flash_leds("success")
                            last_active_time = time.time() # Reset 30s timeout (No activity)
                        elif warned_depth:
                            speak("Go down lower")
                            flash_leds("error")
                            warned_depth = False
                        
                        stage = "up"
                        warned_height = False

                    # Both arms/legs must be sufficiently bent to be "DOWN"
                    elif angle_L < 80 and angle_R < 80:
                        #Didn't go all the way down
                        if warned_height:
                            speak("Go up higher")
                            flash_leds("error")
                            warned_height = False
                        
                        stage = "down"
                        warned_depth = False
                    
                    # --- INTERMEDIATE / QUALITY CHECK ---
                    # Intermediate up (up 30 deg)
                    elif angle_L > 100 and angle_R > 100 and stage == "down":
                        warned_height = True
                    
                    # Intermediate down (down 30 deg)
                    elif angle_L < 150 and angle_R < 150 and stage == "up":
                        warned_depth = True


                    # Extract pixel coordinates for the elbow/knee (the 'B' joint)
                    L_joint_coords = (int(landmarks[L_B.value].x * width), int(landmarks[L_B.value].y * height))
                    R_joint_coords = (int(landmarks[R_B.value].x * width), int(landmarks[R_B.value].y * height))

                    # Draw Left Angle
                    cv2.putText(image, str(int(angle_L)), 
                                L_joint_coords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                    # Draw Right Angle
                    cv2.putText(image, str(int(angle_R)), 
                                R_joint_coords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            except Exception as e:
                print(f"Error in calculation/rendering: {e}")
                pass
            
            # 5. Performance Metrics
            end_time = time.time()
            fps = 1.0 / (end_time - start_time)
            avg_fps.append(fps)
            if len(avg_fps) > 20: avg_fps.pop(0)
            current_fps = int(sum(avg_fps)/len(avg_fps))

            # 6. TIMEOUT CHECK (30 Seconds)
            if time.time() - last_active_time > 10:
                is_running = False
                speak("Session timeout. Powering down.")

            # --- FINAL RENDERING & DEBUGGING ---
            
            # 1. Dashboard Background
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # 2. Rep data
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, .75, (255,255,255), 2, cv2.LINE_AA)
            
            # 3. Stage data
            cv2.putText(image, 'STAGE', (65,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            # Using str(stage) handles cases where stage is None during startup
            cv2.putText(image, str(stage), (60,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)

            # 4. Right-aligned FPS text
            # width is defined at the start of generate_frames
            cv2.putText(image, f"FPS: {current_fps}", (width - 120, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
            # 5. Render detections (The Skeleton)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                )
            
            # 6. Watchdog Timer (Bottom Right)
            # Shows how many seconds the Pi has been 'searching'
            timer_text = f"IDLE: {time_since_detection:.1f}s / 10s"
            cv2.putText(image, timer_text, (width - 250, height - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', image)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    # This HTML points to the /video_feed route you already created
    return "<html><body style='margin:0; background:black;'><img src='/video_feed' width='100%'></body></html>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)