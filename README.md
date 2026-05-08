# Edge Reps AI Fitness Tracker — AI Exercise Detection and Rep Counting

A two-tier edge AI system for real-time exercise classification, rep counting, and form feedback with no cloud connectivity.

## Video Link

https://youtu.be/hWciSzjnYwE

## How It Works
### Laptop bridge

Install dependencies:
```bash
pip install paho-mqtt pyserial
```

Update the broker IP in `bridge.py`:
```python
MQTT_BROKER = "<pi-ip-address>"
```

Close Arduino IDE, then run:
```bash
python bridge.py
```

## How It Works

### ESP32S3 classifier
- Captures QVGA frames from OV2640 camera at ~1.2 FPS
- Resizes to 96x96 and runs MobileNetV1 0.1 INT8 classifier
- 3 classes: `pushup_position`, `squat_position`, `neither`
- 5 consecutive frames above 0.85 confidence threshold required before trigger fires
- Sends `TRIGGER:PUSHUP`, `TRIGGER:SQUAT`, or `TRIGGER:NONE` over USB serial

### bridge.py
- Reads serial triggers from ESP32S3 on COM3
- Converts to JSON MQTT payloads and publishes to `workout/control` topic
- Runs on a laptop connected to the same WiFi network as the Pi

### Raspberry Pi 5 pipeline
- Subscribes to `workout/control` MQTT topic
- Activates MediaPipe Pose on START trigger, returns to idle on STOP or after 10s inactivity
- Calculates bilateral joint angles:
  - Pushup: shoulder → elbow → wrist (up >160°, down <80°)
  - Squat: hip → knee → ankle (up >160°, down <80°)
- Counts reps using FSM (up → down → up = 1 rep)
- Provides form correction cues: "go down lower" / "go up higher"
- Green LED + Voiced rep number on good rep
- Red LED on form error
- Live skeleton overlay streamed via Flask on port 5000

## Model Performance

| Metric | Value |
|---|---|
| Overall accuracy | 93.1% |
| Neither accuracy | 94.9% |
| Pushup accuracy | 81.8% |
| Squat accuracy | 93.2% |
| F1 score | 0.93 |
| AUC-ROC | 0.98 |
| DSP latency | 4 ms |
| Classification latency | 791 ms |
| Total inference latency | 795 ms |
| FPS | 1.20 |
| ESP32S3 power draw | ~300 mW |

## Power Analysis

| Mode | Estimated Power | Energy (3 min) |
|---|---|---|
| Baseline (always active) | ~9.6 W | 1728 J |
| Trigger-gated (optimised) | ~6.3 W avg | 1135 J |
| ESP32S3 constant overhead | ~300 mW | 54 J |
| Total savings | 3.3 W | ~539 J (31%) |

## Known Limitations

- WiFi connectivity on ESP32S3 unreliable — USB serial bridge via laptop used as workaround
- Single exercise per session — switching exercises requires restarting the pipeline
- ~4 second detection latency when switching positions due to 5-frame debounce

## References

1. V. Bazarevsky et al., "BlazePose: On-device real-time body pose tracking," arXiv:2006.10204, 2020.
2. C. Lugaresi et al., "MediaPipe: A framework for building perception pipelines," arXiv:1906.08172, 2019.
3. Z. Cao et al., "Realtime multi-person 2D pose estimation using part affinity fields," arXiv:1611.08050, 2017.
4. J. Stenum et al., "Applications of pose estimation in human health and performance across the lifespan," Sensors, 2021.
5. Real-time human pose estimation using MediaPipe, IEEE Xplore, 2024.
