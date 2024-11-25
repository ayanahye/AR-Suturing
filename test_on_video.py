import cv2
import os
import json
import tempfile
from PIL import Image, ImageDraw
from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
PROJECT_ID = "segmentation-8ybb0"
MODEL_VERSION = "1"

if not API_KEY:
    raise ValueError("Create your own API key from roboflow")

rf = Roboflow(api_key=API_KEY)
project = rf.workspace().project(PROJECT_ID)
model = project.version(MODEL_VERSION).model

video_path = "suturing-real.mp4"
output_folder = "output_frames2"
os.makedirs(output_folder, exist_ok=True)

def process_frame(frame, frame_number):
    # convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # save frame as temp file (needed for API)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        pil_image.save(temp_file, format='JPEG')
        temp_file_path = temp_file.name

    # predict on the whole frame
    result = model.predict(temp_file_path, confidence=10, overlap=50).json()

    # draw predictions
    draw = ImageDraw.Draw(pil_image)
    for pred in result.get("predictions", []):
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

    # delete temp file
    os.remove(temp_file_path)

    return result.get("predictions", [])

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
    print(f"Processed frame {frame_count}")
    frame_count += 1

cap.release()

inference_res = "inference_result2.json"
with open(inference_res, "w") as output_file:
    json.dump(all_results, output_file, indent=4)