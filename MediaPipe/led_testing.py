from pi5neo import Pi5Neo
import time

# LED configuration
# Pi5Neo defaults to /dev/spidev0.0 which is GPIO 10 (Pin 19)
NUM_LEDS = 20

print("Initializing Pi5Neo on GPIO 10 (Pin 19)...")

# Use the context manager to ensure safe cleanup
with Pi5Neo('/dev/spidev0.0', NUM_LEDS) as neo:
    print("Red")
    neo.fill_strip(255, 0, 0)
    neo.update_strip()
    time.sleep(1)

    print("Green")
    neo.fill_strip(0, 255, 0)
    neo.update_strip()
    time.sleep(1)

    print("Blue")
    neo.fill_strip(0, 0, 255)
    neo.update_strip()
    time.sleep(1)

    print("Cleanup (Off)")
    neo.clear_strip()
    neo.update_strip()

print("Test complete!")