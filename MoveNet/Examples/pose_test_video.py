import sys
import os

# 1. Tell Python where to find your parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
import numpy as np
import imageio
import model as m

# 1. Load Video/GIF
if not os.path.exists("dance.gif"):
    os.system("wget -q -O dance.gif https://github.com/tensorflow/tfjs-models/raw/master/pose-detection/assets/dance_input.gif")

image = tf.image.decode_gif(tf.io.read_file('dance.gif'))
num_frames, image_height, image_width, _ = image.shape
crop_region = m.init_crop_region(image_height, image_width)

output_images = []
print(f"Processing {num_frames} frames...")

for frame_idx in range(num_frames):
    # Run inference with cropping logic
    keypoints = m.run_inference_video(
        m.movenet, image[frame_idx], crop_region, 
        crop_size=[m.input_size, m.input_size]
    )
    
    # Visualize the frame
    # We use a 300px height for the output to save RAM on the Pi
    output_images.append(m.draw_prediction_on_image(
        image[frame_idx].numpy().astype(np.int32), 
        keypoints
    ))
    
    # Update crop for next frame
    crop_region = m.determine_crop_region(keypoints, image_height, image_width)
    if frame_idx % 10 == 0:
        print(f"Frame {frame_idx}/{num_frames} complete")

# 2. Save result
imageio.mimsave('dance_output.gif', output_images, duration=100)
print("Success! Result saved as dance_output.gif")