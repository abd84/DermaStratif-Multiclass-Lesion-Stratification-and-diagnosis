import os
import torch
from flask import Flask, request, render_template, url_for
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import math
from torch import nn


# Flask app initialization
app = Flask(__name__)

# Device setup
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Define LoRA
class LoRA(nn.Module):
    def __init__(self, in_features, r=8, alpha=32):
        super(LoRA, self).__init__()
        self.down_proj = nn.Linear(in_features, r, bias=False)
        self.up_proj = nn.Linear(r, in_features, bias=False)
        self.scaling = alpha / r

        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x):
        return self.up_proj(self.down_proj(x)) * self.scaling

# Define the model architecture with LoRA
class EfficientNetWithLoRA(nn.Module):
    def __init__(self, base_model, num_classes, r=8, alpha=32):
        super(EfficientNetWithLoRA, self).__init__()
        self.features = base_model.features  # Pre-trained EfficientNet features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            LoRA(1280, r=r, alpha=alpha),  # LoRA applied here
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x

# Load the model
def load_model():
    base_model = efficientnet_b0(weights=None)
    num_classes = 8  # Replace with the number of classes
    model = EfficientNetWithLoRA(base_model, num_classes=num_classes, r=8, alpha=32)

    # Load the saved state dictionary
    saved_model_path = '/Users/abdullah/Desktop/VS/project/Saved Models/best_model1_lora.pth'
    if os.path.exists(saved_model_path):
        state_dict = torch.load(saved_model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)  # Allow partial loading
        print(f"Model loaded successfully from {saved_model_path}")
    else:
        print(f"Saved model not found at {saved_model_path}.")
        exit()

    model.to(device)
    model.eval()
    return model

model = load_model()

# Define label-to-disease mapping
label_to_disease = {
    0: "Melanoma",
    1: "Basal Cell Carcinoma",
    2: "Squamous Cell Carcinoma",
    3: "Actinic Keratosis",
    4: "Benign Keratosis",
    5: "Dermatofibroma",
    6: "Vascular Lesion",
    7: "Melanocytic Nevus"
}

# Image preprocessing transformations
IMG_HEIGHT, IMG_WIDTH = 224, 224
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Disease information
# Disease information
disease_info = {
    "Melanoma": {
        "severity": "High",
        "symptoms": "Dark, irregularly shaped moles that may itch, bleed, or change over time. It can also appear as a new pigmented or unusual growth on the skin.",
        "treatment": "Treatment includes surgery to remove the lesion, immunotherapy to strengthen the immune system, and targeted therapies aimed at specific genetic changes.",
        "image": "melanoma.png"
    },
    "Melanocytic Nevus": {
        "severity": "Low",
        "symptoms": "Commonly known as moles, these are small, pigmented spots on the skin that are usually harmless. They may be flat or raised and vary in color.",
        "treatment": "Generally, no treatment is needed unless the mole shows signs of changes such as size, shape, or color, in which case medical evaluation is recommended.",
        "image": "melanocytic_nevus.png"
    },
    "Basal Cell Carcinoma": {
        "severity": "Moderate",
        "symptoms": "A pearly or waxy bump on the skin, often with visible blood vessels. It may also appear as a flat, flesh-colored lesion that develops slowly over time.",
        "treatment": "Treatment typically involves surgical excision, cryotherapy, or topical medications. Radiation therapy may be used in some cases.",
        "image": "basal_cell_carcinoma.png"
    },
    "Actinic Keratosis": {
        "severity": "Moderate to High",
        "symptoms": "Rough, scaly patches of skin that may be red, pink, or skin-colored. Commonly found on sun-exposed areas like the face, hands, and scalp.",
        "treatment": "Early treatment includes cryotherapy, topical medications, or laser therapy to prevent progression into squamous cell carcinoma.",
        "image": "actinic_keratosis.png"
    },
    "Benign Keratosis": {
        "severity": "Low",
        "symptoms": "Non-cancerous skin growths such as seborrheic keratosis, solar lentigo, or lichen planus-like keratosis. They are often age-related and appear as brown or black patches.",
        "treatment": "No treatment is necessary unless they become bothersome. Cosmetic removal can be done via cryotherapy or laser surgery.",
        "image": "benign_keratosis.png"
    },
    "Dermatofibroma": {
        "severity": "Low",
        "symptoms": "Small, firm, raised nodules that are typically reddish-brown. Often caused by minor injuries such as insect bites or scratches.",
        "treatment": "Completely harmless and rarely requires treatment. Surgical removal can be considered if it causes discomfort.",
        "image": "dermatofibroma.png"
    },
    "Vascular Lesion": {
        "severity": "Low to Moderate",
        "symptoms": "Marks or growths caused by abnormal blood vessels, such as cherry angiomas, hemangiomas, or port-wine stains. These may vary in size and color.",
        "treatment": "Often no treatment is needed. Cosmetic treatments include laser therapy or sclerotherapy for larger lesions.",
        "image": "vascular_lesion.png"
    },
    "Squamous Cell Carcinoma": {
        "severity": "High",
        "symptoms": "A firm, red nodule or a flat lesion with a scaly, crusted surface. It may grow and spread to other parts of the body if untreated.",
        "treatment": "Surgical removal is the primary treatment. Radiation therapy or topical chemotherapy may be used for advanced cases.",
        "image": "squamous_cell_carcinoma.png"
    }
}


# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Route for prediction
import numpy as np

from skimage.filters import sobel
from skimage.color import rgb2gray
import numpy as np

def calculate_entropy(probabilities):
    """Calculate the entropy of the probability distribution."""
    return -np.sum(probabilities * np.log(probabilities + 1e-8))

def is_valid_skin_image(image_array):
    """Perform robust checks to determine if the image is likely to be valid skin-related."""
    # Convert to grayscale for edge detection
    grayscale_image = rgb2gray(image_array)
    
    # Check edge details using Sobel filter
    edge_map = sobel(grayscale_image)
    edge_density = np.mean(edge_map > 0.1)  # Fraction of significant edges
    
    # Check color distribution
    mean_pixel = np.mean(image_array, axis=(0, 1))  # Mean color per channel
    std_pixel = np.std(image_array, axis=(0, 1))    # Color variance per channel
    
    # Check overall brightness and intensity variance
    brightness = np.mean(grayscale_image)
    variance = np.var(grayscale_image)
    
    # Print metrics for debugging (can be removed in production)
    print(f"Edge Density: {edge_density}")
    print(f"Brightness: {brightness}")
    print(f"Variance: {variance}")
    print(f"Color Std: {np.mean(std_pixel)}")
    
    # Define relaxed thresholds
    EDGE_DENSITY_THRESHOLD = 0.005  # Minimum fraction of edges (relaxed)
    BRIGHTNESS_RANGE = (0.1, 0.9)  # Wider brightness range
    VARIANCE_THRESHOLD = 0.005     # Lower variance threshold
    COLOR_STD_THRESHOLD = 3.0      # Lower color spread requirement
    
    # Validation checks
    is_edge_valid = edge_density > EDGE_DENSITY_THRESHOLD
    is_brightness_valid = BRIGHTNESS_RANGE[0] <= brightness <= BRIGHTNESS_RANGE[1]
    is_variance_valid = variance > VARIANCE_THRESHOLD
    is_color_valid = np.mean(std_pixel) > COLOR_STD_THRESHOLD
    
    # Combine all checks
    return is_edge_valid and is_brightness_valid and is_variance_valid and is_color_valid


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    try:
        # Open and preprocess the image
        image = Image.open(file).convert("RGB")
        image_array = np.array(image)

        # Check if the image passes preprocessing checks
        if not is_valid_skin_image(image_array):
            return render_template(
                'error.html',
                error_message="The uploaded image does not appear to be a valid skin image. Please upload a clear skin image."
            )

        image = transform(image).unsqueeze(0).to(device)

        # Predict the disease
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy()[0]
        predicted_label = np.argmax(probabilities)
        confidence = probabilities[predicted_label]

        # Calculate entropy of the probability distribution
        entropy = calculate_entropy(probabilities)

        # Apply confidence and entropy thresholds
        CONFIDENCE_THRESHOLD = 0.7  # Lower confidence threshold
        ENTROPY_THRESHOLD = 2.0    # Adjusted entropy threshold
        if confidence < CONFIDENCE_THRESHOLD or entropy > ENTROPY_THRESHOLD:
            return render_template(
                'error.html',
                error_message="The uploaded image does not appear to be a valid skin image. Please upload a clear and relevant image."
            )

        # Get predicted disease details
        predicted_disease = label_to_disease.get(predicted_label, "Unknown")
        info = disease_info.get(predicted_disease, {})
        
        # Render the result page with prediction details
        return render_template(
            'result.html',
            predicted_disease=predicted_disease,
            severity=info.get("severity"),
            symptoms=info.get("symptoms"),
            treatment=info.get("treatment"),
            predicted_disease_image=url_for('static', filename=f"images/{info.get('image')}")
        )
    except Exception as e:
        # Handle any unexpected errors
        return render_template(
            'error.html',
            error_message=f"An unexpected error occurred: {str(e)}"
        ), 500





# Run the app
if __name__ == "__main__":
    app.run(debug=True)
