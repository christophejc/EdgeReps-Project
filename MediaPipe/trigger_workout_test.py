import paho.mqtt.client as mqtt
import json
import sys

# Change this to your Pi's IP address if running from a different laptop
MQTT_BROKER = "localhost" 
TOPIC = "workout/control"

client = mqtt.Client()
client.connect(MQTT_BROKER, 1883, 60)

def send_command(action, exercise="PUSHUP"):
    payload = json.dumps({"action": action, "exercise": exercise})
    client.publish(TOPIC, payload)
    print(f"Sent: {payload}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python trigger_workout.py [START/STOP] [PUSHUP/SQUAT]")
    else:
        cmd = sys.argv[1].upper()
        ex = sys.argv[2].upper() if len(sys.argv) > 2 else "PUSHUP"
        send_command(cmd, ex)