from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO("models/best.pt")

# Official CottonWeedID class list
class_names = [
    "carpetweed", "chickweed", "eclipta", "goosegrass", "morningglory",
    "palmeramaranth", "purslane", "ragweed", "sicklepod", "spottedspurge",
    "velvetleaf", "grass"
]

# Weed information dictionary
weed_info = {
    "carpetweed": {"damage": "Competes for nutrients, water, sunlight.", "herbicide": "Pendimethalin / Sethoxydim", "details": "Low-growing, spreads in sandy soils."},
    "chickweed": {"damage": "Forms dense mats, suppressing crop growth.", "herbicide": "2,4-D / Metribuzin", "details": "Cool-season annual, small white flowers."},
    "eclipta": {"damage": "Competes strongly for water & nutrients.", "herbicide": "Atrazine / Metribuzin", "details": "Annual broadleaf, hairy stems, spreads quickly."},
    "goosegrass": {"damage": "Reduces crop vigor.", "herbicide": "Oxadiazon / Sethoxydim", "details": "Perennial grass, thrives in compacted soils."},
    "morningglory": {"damage": "Climbs and smothers crops.", "herbicide": "Glyphosate / Dicamba", "details": "Annual vine with trumpet-shaped flowers."},
    "palmeramaranth": {"damage": "Highly competitive; can reduce yield up to 80%.", "herbicide": "Dicamba / 2,4-D (for glyphosate-resistant types)", "details": "Fast-growing annual; prolific seed producer."},
    "purslane": {"damage": "Forms mats, competing for light & nutrients.", "herbicide": "Pendimethalin / 2,4-D", "details": "Prostrate succulent, spreads rapidly."},
    "ragweed": {"damage": "Competes with crops, produces allergenic pollen.", "herbicide": "Atrazine / 2,4-D", "details": "Annual broadleaf, produces many seeds."},
    "sicklepod": {"damage": "Reduces yield by competing for nutrients.", "herbicide": "2,4-D / Glyphosate", "details": "Annual legume; sickle-shaped pods."},
    "spottedspurge": {"damage": "Dense mats compete for nutrients & light.", "herbicide": "Pendimethalin / 2,4-D", "details": "Low-growing succulent with red stems."},
    "velvetleaf": {"damage": "Aggressive competition; reduces yield.", "herbicide": "Glyphosate / Dicamba", "details": "Tall broadleaf; heart-shaped leaves."},
    "grass": {"damage": "Competes for nutrients & sunlight.", "herbicide": "Pendimethalin / Sethoxydim", "details": "Generic grassy weeds, grows rapidly."}
}

def crop_and_encode(img, boxes, scores, class_ids):
    """Crop detected weeds and include info"""
    crops = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        crop_img = img[y1:y2, x1:x2].copy()
        cv2.rectangle(crop_img, (0, 0), (crop_img.shape[1], crop_img.shape[0]), (128, 0, 128), 3)
        _, buffer = cv2.imencode('.jpg', crop_img)
        img_str = base64.b64encode(buffer).decode('utf-8')

        cls_id = int(class_ids[i])
        weed_name = class_names[cls_id] if cls_id < len(class_names) else f"Unknown({cls_id})"
        info = weed_info.get(weed_name, {"damage": "N/A", "herbicide": "N/A", "details": "N/A"})

        crops.append({
            "weed_name": weed_name,
            "confidence": float(scores[i]),
            "image": img_str,
            "damage": info["damage"],
            "herbicide": info["herbicide"],
            "details": info["details"]
        })
    return crops

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/detect_crop', methods=['POST'])
def detect_crop():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model.predict(source=img, imgsz=640, conf=0.25, verbose=False)
    crops = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        # Draw boxes on full image
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(class_ids[i])
            name = class_names[cls_id] if cls_id < len(class_names) else f"Unknown({cls_id})"
            conf = float(scores[i])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(img, f"{name} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        crops = crop_and_encode(img, boxes, scores, class_ids)

    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return jsonify({"image": img_str, "crops": crops})

if __name__ == '__main__':
    app.run(debug=True)
