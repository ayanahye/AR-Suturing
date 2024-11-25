import cv2
import os
import json
import tempfile
from PIL import Image, ImageDraw
from roboflow import Roboflow
from dotenv import load_dotenv
import concurrent.futures

load_dotenv()

API_KEY = os.getenv("API_KEY")
PROJECT_ID = "wound-detector-bc8ds"
MODEL_VERSION = "1"

if not API_KEY:
    raise ValueError("Create your own API key from roboflow")

rf = Roboflow(api_key=API_KEY)
project = rf.workspace().project(PROJECT_ID)
model = project.version(MODEL_VERSION).model

video_path = "suturing-real.mp4"
output_folder = "output_frames1"
os.makedirs(output_folder, exist_ok=True)

'''
Problem:
    - The model only detects half the cut when passing in the full frame

Approach:
    - Try to tile the image and pass in parts of the frame to detect the cut

Future
    - need to define specific range rather than tiling whole image

Problem: 
    - API requires a file passed in

Approach:
    - keep creating temp files locally...

Problem:
    - This tiling approach is too slow

Approach:
    - Try to use multithreading
'''

def tile_image(image, tile_size=(640, 640), overlap=0.2):
    width, height = image.size
    tiles = []
    # smaller overlapping tiles
    step_size = int(tile_size[0] * (1 - overlap))
    
    # create each tile
    for i in range(0, width, step_size):
        for j in range(0, height, step_size):
            box = (i, j, min(i + tile_size[0], width), min(j + tile_size[1], height))
            tile = image.crop(box)
            tiles.append((tile, box))
    
    return tiles

def process_tile(tile_data):
    tile, box = tile_data
    
    # save tile as temp file (needed for api)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        tile.save(temp_file, format='JPEG')
        temp_file_path = temp_file.name
    
    # lower confidence level to get more of the cut
    result = model.predict(temp_file_path, confidence=30, overlap=50).json()
    
    # adjust prediction positons to match their original positions in image
    for pred in result.get("predictions", []):
        pred["x"] += box[0]
        pred["y"] += box[1]
    
    # delete temp file
    os.remove(temp_file_path)
    return result.get("predictions", [])

def process_frame(frame, frame_number):
    # convert frame to RGB and split it into tiles
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    tiles = tile_image(pil_image)

    # multithreading to process each tile concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_tile, tiles))

    all_predictions = [pred for sublist in results for pred in sublist]
    
    # draw predictions 
    draw = ImageDraw.Draw(pil_image)
    for pred in all_predictions:
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        confidence = pred["confidence"]

        left = x - w / 2
        right = x + w / 2
        top = y - h / 2
        bottom = y + h / 2

        draw.rectangle([left, top, right, bottom], outline="green", width=3)
        draw.text((left, top - 10), f"{confidence:.2f}", fill="green")

    output_path = os.path.join(output_folder, f"frame_{frame_number:04d}.jpg")
    pil_image.save(output_path)
    
    return all_predictions

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
max_frames = int(5 * fps)  
all_results = []

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    result = process_frame(frame, frame_count)
    all_results.append(result)
    print(f"processed frame {frame_count}")
    frame_count += 1

cap.release()

inference_res = "inference_result.json"
with open(inference_res, "w") as output_file:
    json.dump(all_results, output_file, indent=4)