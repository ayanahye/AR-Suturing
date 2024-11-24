import os
import json
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from roboflow import Roboflow

load_dotenv()

# inference SDK has a batch inference option but we will need to do it in real time upon receiving the frame
API_KEY = os.getenv("API_KEY")
folder = "Cut"

# check if the API key is provided
if not API_KEY:
    raise ValueError("Create your own API key from roboflow")

rf = Roboflow(api_key=API_KEY)
project = rf.workspace().project("wound-detector-bc8ds")
model = project.version(1).model

# arbitrary files in Cut
image_files = [os.path.join(folder, f) for f in os.listdir(folder)]
image_files = image_files[:5]
all_results = []

for image_path in image_files:
    # run prediction on the image confidence threshold of 50
    result = model.predict(image_path, confidence=50, overlap=30).json()
    print("results:", result)
    all_results.append(result)

    preds = result.get("predictions", [])

    if preds:
        with Image.open(image_path) as img:
            draw = ImageDraw.Draw(img)
            for pred in preds:
                x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                confidence = pred["confidence"]

                # box coords for lines
                left = x - w / 2
                right = x + w / 2
                top = y - h / 2
                bottom = y + h / 2

                draw.rectangle([left, top, right, bottom], outline="green", width=3)
                draw.text((left, top - 10), f"{confidence:.2f}", fill="green")

            detected_folder = "detected_Cut"
            os.makedirs(detected_folder, exist_ok=True)
            output_path = os.path.join(detected_folder, os.path.basename(image_path))
            img.save(output_path)

inference_res = "inference_result.json"

with open(inference_res, "w") as output_file:
    json.dump(all_results, output_file, indent=4)