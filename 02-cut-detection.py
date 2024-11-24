import os
import json
import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw

load_dotenv()

# inference SDK has a batch inference option but we will need to do it in real time upon recieving the frame

API_KEY = os.getenv("API_KEY")
API_URL = "https://detect.roboflow.com/wound-detector-bc8ds/1"
folder = "Cut"

# arbitrary files in Cut
image_files = [os.path.join(folder, f) for f in os.listdir(folder)]
image_files = image_files[:5]
all_results = []

if not API_KEY:
    raise ValueError("Create your own API key from roboflow")

for image_path in image_files:
    with open(image_path, "rb") as image_file:
        response = requests.post(
            f"{API_URL}?api_key={API_KEY}",
            files={"file": image_file}
        )

    if response.status_code == 200:
        result = response.json()
        print("results:", result)

        all_results.append(result)

        preds = result.get("predictions", [])

        if preds:

            with Image.open(image_path) as img:
                draw = ImageDraw.Draw(img)
                for pred in preds:
                    x,y,w,h = pred["x"], pred["y"], pred["width"], pred["height"]
                    confidence = pred["confidence"]

                    # box coords for lines
                    left = x - w/2
                    right = x + w/2
                    top = y - h/2
                    bottom = y + h/2

                    draw.rectangle([left, top, right, bottom], outline="green", width=3)

                    draw.text((left, top - 10), f"{confidence:.2f}", fill="green")

                detected_folder = "detected_Cut"
                os.makedirs(detected_folder, exist_ok=True)
                output_path = os.path.join(detected_folder, os.path.basename(image_path))
                img.save(output_path)

    else:
        print("Error:", response.status_code, response.text)

inference_res = "inference_result.json"


with open(inference_res, "w") as output_file:
    json.dump(all_results, output_file, indent=4)