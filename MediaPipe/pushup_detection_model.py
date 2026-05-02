from flask import Flask, Response
import cv2
import time
import mediapipe as mp
import numpy as np

# These are from your tutorial - keeping them so your setup is ready for later
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

app = Flask(__name__)

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

def generate_frames():
    # Setup Video Feed
    cap = cv2.VideoCapture(0)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #pushup counter
    counter = 0
    stage = None

    #Performance variables
    avg_fps = []

    #Mediapipe Instance (CHECK WITH THE CONFIDENCE! Currently @ 0.5)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()

            #Recolor image (RGB -> BGR)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            #Make detection
            results = pose.process(image)

            #Recolor Back 
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            #Extracting Landmarks
            try: 
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates (Left/Right)
                shoulder_L = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_L = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_L = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                shoulder_R = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow_R = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist_R = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                # Calculate angle (Left/Right)
                angle_L = calculate_angle(shoulder_L, elbow_L, wrist_L)
                angle_R = calculate_angle(shoulder_R, elbow_R, wrist_R)
                
                # Visualize angle
                cv2.putText(image, str(angle_L), 
                            tuple(np.multiply(elbow_L, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle_R), 
                            tuple(np.multiply(elbow_R, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                # PUSH UP counter logic
                if angle_L > 160 and angle_R > 160 and stage == None: #Starting at UP
                    stage = "up"
                elif angle_L < 80 and angle_R < 80 and stage =='up': # Up to down
                    stage="down"
                elif angle_L > 160 and angle_R > 160 and stage =='down': #Down to up (counter ++)
                    stage="up"
                    counter +=1
                    print(counter)

            
            except:
                pass

            # Performance Metrics
            end_time = time.time()
            fps = 1.0 / (end_time - start_time)
            avg_fps.append(fps)
            if len(avg_fps) > 20: avg_fps.pop(0)
            current_fps = int(sum(avg_fps)/len(avg_fps))

            # Rendering Rep Counter
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
            # Rep data
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (65,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (60,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)

            # Right-aligned FPS text
            cv2.putText(image, f"FPS: {current_fps}", (width - 120, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
            # Render detections 
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )    

            # Encode the frame so it can be sent over the network
            ret, buffer = cv2.imencode('.jpg', image)
            frame_bytes = buffer.tobytes()
            
            # This "yield" acts like a continuous loop for the browser
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


        cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<html><body style='margin:0; background:black;'><img src='/video_feed' width='100%'></body></html>"

if __name__ == "__main__":
    # Runs the server on your local network
    app.run(host='0.0.0.0', port=5000)