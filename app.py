from flask import Flask, request, jsonify, send_from_directory
import torch
from torchvision import transforms
from PIL import Image
import io, base64, os
import torchvision.models.densenet  # Import DenseNet for unpickling

# Add DenseNet to safe globals
torch.serialization.add_safe_globals([torchvision.models.densenet.DenseNet])

# Load model and transform
try:
    model_dict = torch.load("model_with_transforms.pkl", map_location="cpu", weights_only=False)
    model = model_dict["model"]
    transform = model_dict["transform"]
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

model.eval()
class_names = ["Chickenpox", "Measles", "Monkeypox", "Normal"]

# Flask app
app = Flask(
    __name__,
    static_folder="../frontend/static",      # Serve static files (like images)
    template_folder="../frontend"            # For index.html
)

# Helper: Convert PIL Image to Base64 string
def pil_to_base64(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

# ------------------------
# Prediction Route
# ------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file"})

    file = request.files["file"] #Flask retrieves the uploaded image:
    image = Image.open(file.stream).convert("RGB") #Flask retrieves the uploaded image:
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor) # Model Prediction
        probs = torch.nn.functional.softmax(outputs, dim=1)[0] #DenseNet returns logits → converted to softmax probabilities.
                                                                 #It finds the highest probability class and returns results
        pred_idx = torch.argmax(probs).item() #Model Prediction

    return jsonify({
        "prediction": class_names[pred_idx],
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    })

# ------------------------
# Preview Route
# ------------------------
@app.route("/preview", methods=["POST"])
def preview():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(file.stream).convert("RGB")

    pipelines = {
        "normal": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]),
        "train": transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]),
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]),
        "strong": transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomGrayscale(p=0.2),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]),
    }

    preview_data = {}
    for name, pipeline in pipelines.items():
        imgs = []
        current = image.copy()

        for t in pipeline.transforms:
            try:
                current = t(current)
                if isinstance(current, torch.Tensor):
                    current = transforms.ToPILImage()(current)
                imgs.append(pil_to_base64(current))
            except Exception as e:
                print(f"Transform error in {name}: {e}")

        preview_data[name] = imgs

    return jsonify(preview_data)

# ------------------------
# Frontend Route (index.html)
# ------------------------
@app.route("/")
def index():
    return send_from_directory(app.template_folder, "index.html")

# ------------------------
# Route to serve other frontend files (e.g., script.js)
# ------------------------
@app.route("/<path:path>")
def frontend_files(path):
    return send_from_directory(app.template_folder, path)

# ------------------------
# Route to serve static images like accuracy.png, etc.
# ------------------------
@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

# ------------------------
# Run the App
# ------------------------
if __name__ == "__main__":
    app.run(debug=True)
