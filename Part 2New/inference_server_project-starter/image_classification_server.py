from flask import Flask, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import io, signal, sys

# Initialize the Flask app
app = Flask(__name__)

# Load the pretrained model (for example, ResNet18)
# https://pytorch.org/vision/0.8/models.html
# model = models.resnet152(pretrained=true)
# Load the model with default pretrained weights
model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
model.eval()

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the label mapping (for ImageNet classes)
with open("imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if an image file is present in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Read the image file
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Preprocess the image
        input_tensor = transform(image).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_prob, predicted_idx = probabilities.max(0)
        
        # Get the top 5 probabilities and their corresponding indices
        top_probs, top_indices = torch.topk(probabilities, 5)
        top_probs = top_probs.numpy()
        top_indices = top_indices.numpy()
        top_classes = [classes[idx] for idx in top_indices]

        # Format the top 5 results
        results = []
        for i in range(5):
            results.append({
                "class": top_classes[i],
                "probability": top_probs[i].item()
            })
        # Return the top 5 results
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def handle_exit(signal, frame):
    print("Shutting down gracefully...")
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    app.run(host='0.0.0.0', port=5050)

