import cv2
import numpy as np
import os

# convert rgb to hsv
def rgb_to_hsv(r, g, b):
    # std formula
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    diff = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/diff) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/diff) + 120) % 360
    else:
        h = (60 * ((r-g)/diff) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (diff/mx) * 100
    v = mx * 100
    # hsv
    return (h/2, s*2.55, v*2.55)  

def color_segment(frame):
    # convert frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # convert RGB values to HSV for elements based on online color picker
    gloves_hsv = rgb_to_hsv(197, 204, 192)
    clothes_hsv = rgb_to_hsv(87, 128, 189)
    cut_hsv = rgb_to_hsv(204, 128, 132)
    tools_light_hsv = rgb_to_hsv(205, 252, 252)
    tools_dark_hsv = rgb_to_hsv(70, 76, 75)
    skin_hsv = rgb_to_hsv(205, 182, 157)

    # define color ranges around the provided hsv value
    def create_range(hsv, h_tol=10, s_tol=50, v_tol=50):
        return (
            np.array([max(0, hsv[0]-h_tol), max(0, hsv[1]-s_tol), max(0, hsv[2]-v_tol)]),
            np.array([min(180, hsv[0]+h_tol), min(255, hsv[1]+s_tol), min(255, hsv[2]+v_tol)])
        )

    # hsv ranges for each component
    gloves_range = create_range(gloves_hsv)
    clothes_range = create_range(clothes_hsv)
    cut_range = create_range(cut_hsv)
    tools_light_range = create_range(tools_light_hsv, s_tol=20, v_tol=20)
    tools_dark_range = create_range(tools_dark_hsv, s_tol=20, v_tol=20)
    skin_range = create_range(skin_hsv)

    # binary masks for each element -- isolate specific parts of image based on color white if match, black if no match for each component
    gloves_mask = cv2.inRange(hsv, gloves_range[0], gloves_range[1])
    clothes_mask = cv2.inRange(hsv, clothes_range[0], clothes_range[1])
    cut_mask = cv2.inRange(hsv, cut_range[0], cut_range[1])
    tools_light_mask = cv2.inRange(hsv, tools_light_range[0], tools_light_range[1])
    tools_dark_mask = cv2.inRange(hsv, tools_dark_range[0], tools_dark_range[1])
    skin_mask = cv2.inRange(hsv, skin_range[0], skin_range[1])

    # combine light and dark tool masks else bad detection (light reflection)
    tools_mask = cv2.bitwise_or(tools_light_mask, tools_dark_mask)

    # blank image
    result = np.zeros_like(frame)
    # fill in result image with colors based on the masks
    result[gloves_mask > 0] = [255, 255, 0]  
    result[clothes_mask > 0] = [255, 0, 0]    
    result[cut_mask > 0] = [0, 0, 255]        
    result[tools_mask > 0] = [0, 255, 0]      
    result[skin_mask > 0] = [255, 0, 255]     

    return result

segmented_folder = 'segmented_frames'
original_folder = 'original_frames'
os.makedirs(segmented_folder, exist_ok=True)
os.makedirs(original_folder, exist_ok=True)

# open the video file
video = cv2.VideoCapture('suturing-real.mp4')

frame_count = 0
while True:
    # read a frame from the video
    ret, frame = video.read()
    if not ret:
        break  # exit loop if no more frames
    
    # process the frame using color segmentation
    segmented_frame = color_segment(frame)
    
    # save every 10th segmented frame
    if frame_count % 10 == 0:
        segmented_output_path = os.path.join(segmented_folder, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(segmented_output_path, segmented_frame)
    
    # save the first 100 original frames
    if frame_count < 100:
        original_output_path = os.path.join(original_folder, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(original_output_path, frame)
    
    frame_count += 1
    
    # display the segmented frame
    cv2.imshow('segmented frame', segmented_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # exit loop if q is pressed

video.release()
cv2.destroyAllWindows()