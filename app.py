from flask import Flask, render_template, request
import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

app = Flask(__name__)          # ✅ FIXED
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# List of brands to detect in text
electronics_brands = ['iphone', 'apple', 'samsung', 'oneplus', 'vivo', 'oppo', 'mi', 'redmi', 'xiaomi', 'sony']


def detect_brand(text):
    """Extract brand name from text or filename."""
    if not text:
        return None
    text = text.lower().replace(" ", "")
    for brand in electronics_brands:
        if brand in text:
            return brand
    return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    description = request.form.get('description', '').strip().lower()

    if not file or file.filename == "":
        return "No file selected", 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Brand detection
    brand_from_image = detect_brand(file.filename)
    brand_from_desc = detect_brand(description)

    # If brands differ → reject immediately
    if brand_from_image and brand_from_desc:
        if brand_from_image != brand_from_desc:
            return render_template(
                "result.html",
                image=filepath,
                desc=description,
                result="❌ INCORRECT — Brand mismatch"
            )

    # CLIP matching
    image = Image.open(filepath).convert("RGB")

    # Stronger prompt set
    prompts = [
        description,
        f"a photo of {description}",
        f"the product {description}",
        f"{description} mobile",
        f"{description} smartphone",
        f"original {description} device",
        f"clear product photo of {description}",
        f"real {description} image"
    ]

    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)

    image_emb = outputs.image_embeds
    text_emb = outputs.text_embeds
    sims = torch.cosine_similarity(image_emb, text_emb)

    best_score = float(sims.max())

    # THRESHOLD FIX (Watermark-safe)
    if best_score >= 0.28:
        result = "✅ CORRECT — Image matches description"
    else:
        result = "❌ INCORRECT — Image does NOT match description"

    return render_template(
        "result.html",
        image=filepath,
        desc=description,
        result=result + f"<br>Score: {best_score:.3f}"
    )


if __name__ == "__main__":    # ✅ FIXED
    app.run(host="127.0.0.1", port=5000, debug=True)
