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
import datetime
import subprocess
import atexit
import psutil

# --- GLOBAL STATE & ORIGINAL CONFIG ---
app = Flask(__name__)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

is_running = False
current_exercise = "PUSHUP"
last_active_time = time.time()
counter = 0
stage = None
latest_frame = None 

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

# --- HARDWARE HELPERS ---
def speak(text):
    """
    Non-blocking speech using threading and subprocess to 
    target the USB hardware directly.
    """
    def target():
        try:
            # We use the exact hardware address (hw:2,0) and 
            # silence stderr to keep your console clean.
            subprocess.run(
                ['espeak-ng', text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"🔊 Audio Thread Error: {e}")

    # Create and start the thread so the rest of your code doesn't wait for the speech
    threading.Thread(target=target, daemon=True).start()

def check_audio():
    # Keep your specific ID check
    if os.path.exists('/proc/asound/UACDemoV10'):
        print("✅ Audio Hardware: FOUND (UACDemoV10)")
        # Quick test to ensure Card 2 is actually the one
        # This helps if the Pi reorders cards on reboot
        return True
    else:
        print("❌ Audio Hardware: MISSING (Check USB connection)")
        return False

# Run the check
audio_ready = check_audio()


# ---LED CONFIGURATION ---
NUM_LEDS = 200
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


# POWER STATISTICS

class WorkoutStats:
    def __init__(self):
        self.session_fps = []
        self.session_cpu = [psutil.cpu_percent(interval=None)]
        self.start_time = None

    def get_pi_metrics(self):
        """Retrieves hardware stats specifically for Raspberry Pi."""
        try:
            # Measure Core Voltage
            volt = subprocess.check_output(["vcgencmd", "measure_volts", "core"]).decode("utf-8").strip()
            # Measure Clock Speed
            clock = subprocess.check_output(["vcgencmd", "measure_clock", "arm"]).decode("utf-8").strip()
            # Check for Throttling (0x0 means everything is fine)
            throttled = subprocess.check_output(["vcgencmd", "get_throttled"]).decode("utf-8").strip()

            # Capture current CPU usage percentage
            cpu_usage = psutil.cpu_percent()
            
            return {
                "voltage": volt.split('=')[1],
                "clock_speed": f"{int(clock.split('=')[1]) / 10**6:.1f} MHz",
                "throttled_state": throttled.split('=')[1],
                "cpu_usage": f"{cpu_usage}%"
            }
        except Exception:
            return {"error": "Could not access vcgencmd (Check if running on Pi)"}

    def generate_summary(self):
        """Calculates final session stats."""
        if not self.session_cpu:
            return "No hardware data captured during this session."
        
        avg_fps = sum(self.session_fps) / len(self.session_fps) if len(self.session_fps) > 0 else 0.0
        # Calculate average CPU usage over the session
        avg_cpu = sum(self.session_cpu) / len(self.session_cpu) if len(self.session_cpu) > 0 else 0.0

        duration = time.time() - self.start_time
        hw = self.get_pi_metrics()

        summary = (
            f"\n--- SESSION SUMMARY ---\n"
            f"Duration: {duration:.1f}s\n"
            f"Average Performance: {avg_fps:.2f} FPS\n"
            f"Average CPU Usage: {avg_cpu:.1f}%\n" # Added to summary
            f"Final Voltage: {hw.get('voltage')}\n"
            f"CPU Clock: {hw.get('clock_speed')}\n"
            f"Throttled: {hw.get('throttled_state')} (0x0 is ideal)\n"
            f"-----------------------\n"
        )
        return summary

# Initialize globally
stats_tracker = WorkoutStats()
stats_tracker.start_time = time.time()
stats_tracker.session_cpu = []


def background_monitor():
    """Continuously tracks CPU usage regardless of whether an exercise is active."""
    # Initialize the first call to set a baseline
    psutil.cpu_percent(interval=None) 
    
    while True:
        # Use None so it doesn't pause the thread for a full second
        usage = psutil.cpu_percent(interval=None)
        
        # Only record if it's a valid number (sometimes the first call is 0.0)
        stats_tracker.session_cpu.append(usage)
        
        # Frequency of background tracking (e.g., every 0.5 seconds)
        time.sleep(0.5)

# Start the monitor as soon as the script begins
threading.Thread(target=background_monitor, daemon=True).start()


def final_shutdown_report():
    print("\n" + "!"*40)
    print("PROGRAM TERMINATED: FINAL HARDWARE REPORT")
    print(stats_tracker.generate_summary())
    print("!"*40 + "\n")

atexit.register(final_shutdown_report)

# --- THE POWER-SAVING WORKER (Full Logic Restored) ---
def pose_worker():
    global is_running, last_active_time, counter, stage, latest_frame
    
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # --- Initialize Session Stats ---
    stats_tracker.session_fps = []
    avg_fps = []
    
    # RESTORED: Your original flags for intermediate/incomplete reps
    warned_depth = False 
    warned_height = False

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while is_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            start_time = time.time()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            time_since_detection = time.time() - last_active_time

            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    exercise_joints = JOINT_MAP[current_exercise]

                    # RESTORED: Original Angle Calculations
                    L_A, L_B, L_C = exercise_joints["left"]
                    angle_L = calculate_angle(
                        [landmarks[L_A.value].x, landmarks[L_A.value].y],
                        [landmarks[L_B.value].x, landmarks[L_B.value].y],
                        [landmarks[L_C.value].x, landmarks[L_C.value].y]
                    )

                    R_A, R_B, R_C = exercise_joints["right"]
                    angle_R = calculate_angle(
                        [landmarks[R_A.value].x, landmarks[R_A.value].y],
                        [landmarks[R_B.value].x, landmarks[R_B.value].y],
                        [landmarks[R_C.value].x, landmarks[R_C.value].y]
                    )

                    # --- RESTORED: BILATERAL REP LOGIC ---
                    if angle_L > 160 and angle_R > 160:
                        if stage == "down":
                            counter += 1
                            speak(str(counter))
                            flash_leds("success")
                            last_active_time = time.time() 
                        elif warned_depth:
                            speak("Go down lower")
                            flash_leds("error")
                            warned_depth = False
                        stage = "up"
                        warned_height = False

                    elif angle_L < 80 and angle_R < 80:
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
                print(f"Calculation Error: {e}")

            # --- NEW: Record FPS for Summary ---
            loop_time = time.time() - start_time
            if loop_time > 0:
                current_loop_fps = 1.0 / loop_time
                stats_tracker.session_fps.append(current_loop_fps)
                # ADD THIS LINE:
                stats_tracker.session_cpu.append(psutil.cpu_percent())

            # RESTORED: 15s Timeout Logic
            if time.time() - last_active_time > 10:
                is_running = False
                speak(f"Session timeout. You completed {counter} {current_exercise}'s")

            # --- RESTORED: DASHBOARD RENDERING ---
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            cv2.putText(image, 'REPS', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            cv2.putText(image, str(counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, .75, (255,255,255), 2)
            cv2.putText(image, 'STAGE', (65,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            cv2.putText(image, str(stage), (60,60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
            #cv2.putText(image, f"FPS: {current_fps}", (width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
            # 5. Render detections (The Skeleton)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                )

            timer_text = f"IDLE: {time_since_detection:.1f}s / 10s"
            cv2.putText(image, timer_text, (width - 250, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Package for Flask
            ret, buffer = cv2.imencode('.jpg', image)
            latest_frame = buffer.tobytes()

        #Print power statistics
        # --- NEW: Write to file instead of just printing ---
        summary_data = stats_tracker.generate_summary()
        
        with open("workout_history.log", "a") as f:
            # Add a clear timestamp for the entry
            f.write(f"\n--- LOG ENTRY: {time.ctime()} ---\n")
            f.write(summary_data)
            f.write("\n" + "="*40 + "\n")
            
        print("Session data successfully appended to workout_history.log")
        print(summary_data)

    cap.release()
    latest_frame = None

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360-angle if angle > 180.0 else angle

# --- MQTT & FLASK (Standardized for Operational Mode) ---
def on_message(client, userdata, msg):
    global is_running, current_exercise, last_active_time, counter, stage
    try:
        data = json.loads(msg.payload.decode())
        if data.get("action") == "START" and not is_running:
            current_exercise = data.get("exercise", "PUSHUP").upper()
            is_running = True
            last_active_time = time.time()
            counter = 0
            stage = None
            speak(f"Starting {current_exercise} mode")
            threading.Thread(target=pose_worker, daemon=True).start()
        elif data.get("action") == "STOP":
            is_running = False
            speak("Session stopped.")
    except Exception as e:
        print(f"MQTT Error: {e}")

mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.on_message = on_message
mqtt_client.connect("localhost", 1883, 60)
mqtt_client.subscribe("workout/control")
mqtt_client.loop_start()

@app.route('/video_feed')
def video_feed():
    def generate_stream():
        while True:
            if latest_frame:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
            else:
                black_frame = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(black_frame, "WAITING FOR EXERCISE MQTT", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                _, buffer = cv2.imencode('.jpg', black_frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<html><body style='background:black; color:white; text-align:center;'><img src='/video_feed' width='80%'></body></html>"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)