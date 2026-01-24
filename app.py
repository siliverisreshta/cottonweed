from flask import Flask, render_template, request, jsonify,url_for
from ultralytics import YOLO
import cv2
import uuid
import numpy as np
import base64
import os

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO("models/best.pt")

OUTPUT_DIR = "static/output"
CROP_DIR = "static/crops"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)
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

def crop_and_save(img, boxes, scores, class_ids):
    crops = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        crop_img = img[y1:y2, x1:x2].copy()
        crop_img = cv2.resize(crop_img, (320, 320))

        filename = f"{uuid.uuid4()}.jpg"
        path = os.path.join("static/crops", filename)
        cv2.imwrite(path, crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

        cls_id = int(class_ids[i])
        weed_name = class_names[cls_id]
        info = weed_info.get(weed_name, {})

        crops.append({
            "weed_name": weed_name,
            "confidence": float(scores[i]),
            "imageUrl": url_for('static', filename=f"crops/{filename}"),
            "damage": info.get("damage", "N/A"),
            "herbicide": info.get("herbicide", "N/A"),
            "details": info.get("details", "N/A")
        })

    return crops
@app.route('/')
def fet():
    return "Hello"

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/detect_crop', methods=['POST'])
def detect_crop():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    # Read & decode image
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # Resize full image early (CRITICAL)
    max_width = 1280
    if img.shape[1] > max_width:
        ratio = max_width / img.shape[1]
        img = cv2.resize(img, (max_width, int(img.shape[0] * ratio)))

    # Run YOLO
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

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(
                img,
                f"{name} {conf:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2
            )

        # Crop + save weeds (NO base64)
        crops = crop_and_save(img, boxes, scores, class_ids)

    # Save annotated full image
    output_filename = f"{uuid.uuid4()}.jpg"
    output_path = os.path.join("static/output", output_filename)
    cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

    return jsonify({
        "imageUrl": url_for('static', filename=f"output/{output_filename}"),
        "crops": crops
    })

if __name__ == "__main__":
    # Render provides a PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    # Must listen on 0.0.0.0 for external access
    app.run(host='0.0.0.0', port=port)
