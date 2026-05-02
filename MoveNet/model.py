
import tensorflow as tf
#import tensorflow_hub as hub
import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('Agg') # Essential for Raspberry Pi SSH
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import imageio

# --- CONFIGURATION ---
model_name = "movenet_lightning_f16"
input_size = 192

# --- HELPER FUNCTIONS ---
KEYPOINT_DICT = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c', (0, 5): 'm', (0, 6): 'c',
    (5, 7): 'm', (7, 9): 'm', (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm',
    (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm', (12, 14): 'c', (14, 16): 'c'
}

# For video
KEYPOINT_EDGE_INDS_TO_COLOR_BGR = {
    (0, 1): (255, 0, 255), (0, 2): (255, 255, 0), (1, 3): (255, 0, 255),
    (2, 4): (255, 255, 0), (0, 5): (255, 0, 255), (0, 6): (255, 255, 0),
    (5, 7): (255, 0, 255), (7, 9): (255, 0, 255), (6, 8): (255, 255, 0),
    (8, 10): (255, 255, 0), (5, 6): (0, 255, 255), (5, 11): (255, 0, 255),
    (6, 12): (255, 255, 0), (11, 12): (0, 255, 255), (11, 13): (255, 0, 255),
    (13, 15): (255, 0, 255), (12, 14): (255, 255, 0), (14, 16): (255, 255, 0)
}

def _keypoints_and_edges_for_display(keypoints_with_scores, height, width, keypoint_threshold=0.11):
    keypoints_all = []
    keypoint_edges_all = []
    edge_colors = []
    num_instances, _, _, _ = keypoints_with_scores.shape
    for idx in range(num_instances):
        kpts_x = keypoints_with_scores[0, idx, :, 1]
        kpts_y = keypoints_with_scores[0, idx, :, 0]
        kpts_scores = keypoints_with_scores[0, idx, :, 2]
        kpts_absolute_xy = np.stack([width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
        kpts_above_thresh_absolute = kpts_absolute_xy[kpts_scores > keypoint_threshold, :]
        keypoints_all.append(kpts_above_thresh_absolute)

        for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if (kpts_scores[edge_pair[0]] > keypoint_threshold and kpts_scores[edge_pair[1]] > keypoint_threshold):
                line_seg = np.array([kpts_absolute_xy[edge_pair[0]], kpts_absolute_xy[edge_pair[1]]])
                keypoint_edges_all.append(line_seg)
                edge_colors.append(color)
    keypoints_xy = np.concatenate(keypoints_all, axis=0) if keypoints_all else np.zeros((0, 17, 2))
    edges_xy = np.stack(keypoint_edges_all, axis=0) if keypoint_edges_all else np.zeros((0, 2, 2))
    return keypoints_xy, edges_xy, edge_colors

def draw_prediction_on_image(image, keypoints_with_scores):
    height, width, channel = image.shape
    aspect_ratio = float(width) / height
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    fig.tight_layout(pad=0)
    plt.axis('off')
    ax.imshow(image)
    
    line_segments = LineCollection([], linewidths=(4), linestyle='solid')
    ax.add_collection(line_segments)
    scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

    (keypoint_locs, keypoint_edges, edge_colors) = _keypoints_and_edges_for_display(keypoints_with_scores, height, width)

    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
    if keypoint_locs.shape[0]:
        scat.set_offsets(keypoint_locs)

    fig.canvas.draw()
    # Modern Matplotlib buffer handling
    image_from_plot = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return image_from_plot[:, :, :3] # Convert RGBA to RGB

# --- CROPPING FOR VIDEO ---
MIN_CROP_KEYPOINT_SCORE = 0.2

def init_crop_region(image_height, image_width):
    if image_width > image_height:
        box_height, box_width = image_width / image_height, 1.0
        y_min, x_min = (image_height / 2 - image_width / 2) / image_height, 0.0
    else:
        box_height, box_width = 1.0, image_height / image_width
        y_min, x_min = 0.0, (image_width / 2 - image_height / 2) / image_width
    return {'y_min': y_min, 'x_min': x_min, 'y_max': y_min + box_height, 
            'x_max': x_min + box_width, 'height': box_height, 'width': box_width}

def torso_visible(keypoints):
    return ((keypoints[0, 0, 11, 2] > MIN_CROP_KEYPOINT_SCORE or
             keypoints[0, 0, 12, 2] > MIN_CROP_KEYPOINT_SCORE) and
            (keypoints[0, 0, 5, 2] > MIN_CROP_KEYPOINT_SCORE or
             keypoints[0, 0, 6, 2] > MIN_CROP_KEYPOINT_SCORE))

def determine_crop_region(keypoints, image_height, image_width):
    target_keypoints = {i: [keypoints[0, 0, i, 0] * image_height, keypoints[0, 0, i, 1] * image_width] for i in range(17)}
    if torso_visible(keypoints):
        center_y = (target_keypoints[11][0] + target_keypoints[12][0]) / 2
        center_x = (target_keypoints[11][1] + target_keypoints[12][1]) / 2
        
        # Calculate distances to find body range
        distances = []
        for i in range(17):
            if keypoints[0, 0, i, 2] > MIN_CROP_KEYPOINT_SCORE:
                distances.append([abs(center_y - target_keypoints[i][0]), abs(center_x - target_keypoints[i][1])])
        
        max_dist = np.max(distances, axis=0) if distances else [0, 0]
        crop_length_half = np.amax([max_dist[1] * 1.9, max_dist[0] * 1.9])
        
        # Keep crop within image bounds
        tmp = np.array([center_x, image_width - center_x, center_y, image_height - center_y])
        crop_length_half = np.amin([crop_length_half, np.amax(tmp)])
        
        if crop_length_half > max(image_width, image_height) / 2:
            return init_crop_region(image_height, image_width)
        
        crop_length = crop_length_half * 2
        return {
            'y_min': (center_y - crop_length_half) / image_height,
            'x_min': (center_x - crop_length_half) / image_width,
            'y_max': (center_y + crop_length_half) / image_height,
            'x_max': (center_x + crop_length_half) / image_width,
            'height': crop_length / image_height, 'width': crop_length / image_width
        }
    return init_crop_region(image_height, image_width)

def run_inference_video(movenet_fn, image, crop_region, crop_size):
    image_height, image_width, _ = image.shape
    boxes = [[crop_region['y_min'], crop_region['x_min'], crop_region['y_max'], crop_region['x_max']]]
    input_image = tf.image.crop_and_resize(tf.expand_dims(image, axis=0), boxes, [0], crop_size)
    
    keypoints_with_scores = movenet_fn(input_image)
    
    # Map back to original image coordinates
    for idx in range(17):
        keypoints_with_scores[0, 0, idx, 0] = (crop_region['y_min'] * image_height + crop_region['height'] * image_height * keypoints_with_scores[0, 0, idx, 0]) / image_height
        keypoints_with_scores[0, 0, idx, 1] = (crop_region['x_min'] * image_width + crop_region['width'] * image_width * keypoints_with_scores[0, 0, idx, 1]) / image_width
    return keypoints_with_scores
# --- FINISH CROPPING FOR VIDEO ---




# --- SECTION 3: MODEL LOADING ---
if not os.path.exists("model.tflite"):
    print("Downloading MoveNet TFLite model...")
    os.system("wget -q -O model.tflite 'https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite'")

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

def movenet(input_image):
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])
'''
# --- MAIN EXECUTION ---
def main():
    # 1. Get Image
    if not os.path.exists("input_image.jpeg"):
        print("Downloading test image...")
        os.system("curl -o input_image.jpeg https://images.pexels.com/photos/4384679/pexels-photo-4384679.jpeg --silent")
    
    image = tf.image.decode_jpeg(tf.io.read_file('input_image.jpeg'))
    
    # 2. Process
    input_tensor = tf.expand_dims(image, axis=0)
    input_tensor = tf.image.resize_with_pad(input_tensor, input_size, input_size)
    
    print("Running detection...")
    keypoints = movenet(input_tensor)
    
    # 3. Visualize
    display_image = tf.cast(tf.image.resize_with_pad(tf.expand_dims(image, axis=0), 1280, 1280), dtype=tf.int32)
    output_overlay = draw_prediction_on_image(np.squeeze(display_image.numpy(), axis=0), keypoints)
    
    # 4. Save
    plt.imsave('final_result.png', output_overlay)
    print("Success! Open 'final_result.png' to see the result.")

if __name__ == "__main__":
    main()
'''