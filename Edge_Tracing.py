import cv2
import numpy as np
import os

# This is based on masks given in training data not the model prediction
    # for testing purposes

# process each mask to generate edge overlay and coords
def process_mask(mask_path, image_path, output_path):
    # read the binary mask as grayscale and original image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path)
    
    # find contours of the white segments
    # ref: https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
    # RETR_EXTERNAL -- get only the extreme outer contours
    # CHAIN_APPROX_SIMPLE to compress horiz, vert and diagonal segments - only keep end points
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # draw green outline on the original image
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    
    # save 
    cv2.imwrite(output_path, image)
    
    # print coordinates of the edge
    for contour in contours:
        print(f"coordinates for {os.path.basename(mask_path)}:")
        for point in contour:
            x, y = point[0]
            print(f"({x}, {y})", end=" ")
        print("\n")

# set up directories
mask_dir = "train_masks_sample"
image_dir = "train_images_sample"
output_dir = "train_masks_edge_trace"
os.makedirs(output_dir, exist_ok=True)

# for each mask generate the green edge tracing
for mask_file in os.listdir(mask_dir):
    mask_path = os.path.join(mask_dir, mask_file)
    image_path = os.path.join(image_dir, mask_file)
    output_path = os.path.join(output_dir, mask_file)
    process_mask(mask_path, image_path, output_path)

