import sys
import os

# 1. Tell Python where to find your parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import model as m # Import everything from the other file

# 1. Load Input Image
if not os.path.exists("input_image.jpeg"):
    os.system("curl -o input_image.jpeg https://images.pexels.com/photos/4384679/pexels-photo-4384679.jpeg --silent")

image_path = 'input_image.jpeg'
image = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image)

# 2. Run Inference
input_image = tf.expand_dims(image, axis=0)
input_image = tf.image.resize_with_pad(input_image, m.input_size, m.input_size)

# Call the movenet function from model.py
keypoints_with_scores = m.movenet(input_image)

# 3. Visualize
display_image = tf.expand_dims(image, axis=0)
display_image = tf.cast(tf.image.resize_with_pad(display_image, 1280, 1280), dtype=tf.int32)

# Call the drawing function from model.py
output_overlay = m.draw_prediction_on_image(
    np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)

# 4. Save/Show (Save to file for Raspberry Pi)
plt.figure(figsize=(5, 5))
plt.imshow(output_overlay)
plt.axis('off')
plt.savefig('single_inference_result.png')
print("Inference Complete. Result saved as single_inference_result.png")