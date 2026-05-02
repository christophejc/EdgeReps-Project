import serial
import paho.mqtt.client as mqtt
import time

# ── SETTINGS ──────────────────────────────────────────────────────────────────
SERIAL_PORT  = "COM3"
BAUD_RATE    = 115200
MQTT_BROKER  = "172.20.10.9"
MQTT_PORT    = 1883
MQTT_TOPIC   = "workout/control"
# ─────────────────────────────────────────────────────────────────────────────

# connect MQTT
print(f"Connecting to MQTT broker at {MQTT_BROKER}...")
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
    print("MQTT connected")
except Exception as e:
    print(f"MQTT connection failed: {e}")
    print("Check that your partner's Pi is running mosquitto and you are on the same network")
    exit()

print(f"Opening serial port {SERIAL_PORT}...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Serial connected on {SERIAL_PORT}")
except Exception as e:
    print(f"Serial connection failed: {e}")
    print("Make sure Arduino IDE is closed and ESP32S3 is plugged in")
    exit()

print("\nBridge running - waiting for triggers from ESP32S3...")
print("Get into pushup or squat position in front of the camera\n")

while True:
    try:
        line = ser.readline().decode('utf-8').strip()

        if not line:
            continue

        #if "TRIGGER" in line or ">>" in line:
        print(f"ESP32S3: {line}")

        if line == "TRIGGER:PUSHUP":
            payload = '{"action": "START", "exercise": "PUSHUP"}'
            client.publish(MQTT_TOPIC, payload)
            print(f">> MQTT sent: {payload}")

        elif line == "TRIGGER:SQUAT":
            payload = '{"action": "START", "exercise": "SQUAT"}'
            client.publish(MQTT_TOPIC, payload)
            print(f">> MQTT sent: {payload}")

    except UnicodeDecodeError:
        continue
    except KeyboardInterrupt:
        print("\nStopping bridge...")
        ser.close()
        client.loop_stop()
        client.disconnect()
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(0.1)