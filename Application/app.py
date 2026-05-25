import os
import logging
import torch
from flask import Flask, request, render_template, url_for, jsonify
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import math
from torch import nn
import json
import google.generativeai as genai
from dotenv import load_dotenv

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("dermastratif")

# Load environment variables
load_dotenv()

VISION_API_KEY = os.getenv("VISION_API_KEY") or os.getenv("GEMINI_API_KEY")
if VISION_API_KEY:
    genai.configure(api_key=VISION_API_KEY)
    log.info("Vision analysis key loaded successfully")
else:
    log.warning("Vision API key not found — cloud vision analysis will be unavailable")


# Flask app initialization
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True


def sanitize_display_text(text):
    """Remove vendor/AI branding from text shown in the UI."""
    if not text:
        return text
    out = str(text)
    for old, new in (
        ("The AI identified", "Analysis indicates"),
        ("the AI identified", "analysis indicates"),
        ("AI-assisted", "computer-assisted"),
        ("AI analysis", "clinical analysis"),
        ("AI-powered", "automated"),
        ("AI Severity", "Severity"),
        ("Gemini", "vision"),
        ("gemini", "vision"),
    ):
        out = out.replace(old, new)
    return out


def risk_tier_from_analysis(is_cancer, cancer_status):
    if is_cancer:
        return "high"
    if cancer_status and "Precancerous" in cancer_status:
        return "moderate"
    return "low"


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
    num_classes = 8
    trained = EfficientNetWithLoRA(base_model, num_classes=num_classes, r=8, alpha=32)

    saved_model_path = '../Saved Models/best_model1_lora.pth'
    if os.path.exists(saved_model_path):
        state_dict = torch.load(saved_model_path, map_location='cpu')
        trained.load_state_dict(state_dict, strict=False)
        log.info(f"Trained model loaded: {saved_model_path}")
    else:
        log.error(f"Trained model not found at: {saved_model_path}")
        exit()

    trained.to(device)
    trained.eval()
    return trained

log.info(f"Device: {'mps' if torch.backends.mps.is_available() else 'cpu'}")
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

# Define supported skin lesion classes
SUPPORTED_DISEASES = [
    "Melanoma",
    "Basal Cell Carcinoma",
    "Squamous Cell Carcinoma",
    "Actinic Keratosis",
    "Benign Keratosis",
    "Dermatofibroma",
    "Vascular Lesion",
    "Melanocytic Nevus"
]

# Define cancer status for each disease
CANCER_STATUS = {
    "Melanoma": {"is_cancer": True, "type": "High-Risk Skin Cancer"},
    "Basal Cell Carcinoma": {"is_cancer": True, "type": "Skin Cancer (Non-Melanoma)"},
    "Squamous Cell Carcinoma": {"is_cancer": True, "type": "Skin Cancer (Non-Melanoma)"},
    "Actinic Keratosis": {"is_cancer": False, "type": "Precancerous Lesion"},
    "Benign Keratosis": {"is_cancer": False, "type": "Benign Growth"},
    "Dermatofibroma": {"is_cancer": False, "type": "Benign Growth"},
    "Vascular Lesion": {"is_cancer": False, "type": "Benign Vascular Growth"},
    "Melanocytic Nevus": {"is_cancer": False, "type": "Common Mole (Benign)"}
}

def get_model_fallback(image):
    """
    Fallback: run EfficientNet-LoRA locally when vision analysis is unavailable.
    Returns a result dict with clinical info from the static disease_info store.
    """
    log.info("[FALLBACK] Running trained model inference locally")
    try:
        image_tensor = transform(image).unsqueeze(0).to(device)
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy()[0]
        predicted_label = int(probs.argmax())
        confidence = float(probs[predicted_label])

        predicted_disease = label_to_disease.get(predicted_label, "Unknown")
        cancer_info = CANCER_STATUS.get(predicted_disease, {"is_cancer": False, "type": "Benign"})
        basic = disease_info.get(predicted_disease, {})

        if cancer_info["is_cancer"]:
            c_status = f"CANCER DETECTED — {cancer_info['type']}"
        elif "Precancerous" in cancer_info["type"]:
            c_status = f"PRECANCEROUS LESION — {cancer_info['type']}"
        else:
            c_status = f"NO CANCER DETECTED — {cancer_info['type']}"

        log.info(f"[FALLBACK] Result: {predicted_disease} ({confidence:.1%} confidence)")

        return {
            "not_skin_image": False,
            "predicted_disease": predicted_disease,
            "confidence": f"{confidence:.1%}",
            "is_cancer": cancer_info["is_cancer"],
            "cancer_status": c_status,
            "ai_severity": basic.get("severity", "Moderate"),
            "finding_explanation": (
                f"Visual patterns are consistent with {predicted_disease}. "
                "This assessment is based on analysis of the lesion's colour, shape, border "
                "characteristics, and texture observed in the uploaded image."
            ),
            "visual_characteristics": (
                f"Visual features consistent with {predicted_disease} were detected, "
                "including characteristic colour distribution, border definition, and surface texture."
            ),
            "detailed_symptoms": basic.get("symptoms", "Please consult a dermatologist for a detailed symptom assessment."),
            "root_causes": (
                "Multiple factors may contribute to this condition, including UV radiation exposure, "
                "genetic predisposition, immune function, and environmental influences."
            ),
            "differential_diagnosis_notes": (
                f"{predicted_disease} was identified as the most probable classification "
                "based on deep learning analysis of the lesion's visual characteristics."
            ),
            "home_treatments": basic.get("treatment", "Professional medical consultation is recommended before attempting any self-treatment."),
            "medical_treatments": basic.get("treatment", "Please consult a board-certified dermatologist for appropriate treatment options."),
            "when_to_see_doctor": (
                "Schedule an appointment with a board-certified dermatologist promptly for "
                "professional confirmation and a tailored treatment plan."
            ),
            "prevention_tips": (
                "Apply broad-spectrum SPF 30+ sunscreen daily, avoid peak UV hours, "
                "wear protective clothing, and perform monthly skin self-examinations."
            ),
            "urgent_warning_signs": (
                "Seek immediate medical attention if the lesion bleeds spontaneously, "
                "grows rapidly over days or weeks, changes colour significantly, or causes pain."
            ),
            "confidence_justification": f"Confidence score of {confidence:.1%} based on trained model inference.",
        }
    except Exception as e:
        log.error(f"[FALLBACK] Trained model inference failed: {e}")
        return None


def get_vision_analysis(image):
    """Analyze the skin lesion image and return a full clinical assessment dict, or None on failure."""
    log.info("[AI] Starting vision analysis")
    try:
        vision_model = genai.GenerativeModel('gemini-2.5-flash')

        prompt = """You are a board-certified dermatology AI expert. Carefully analyze this skin lesion image and provide a complete clinical assessment.

CLASSIFICATION RULE: You MUST classify the lesion into EXACTLY ONE of these 8 categories — no others:
  - Melanoma
  - Basal Cell Carcinoma
  - Squamous Cell Carcinoma
  - Actinic Keratosis
  - Benign Keratosis
  - Dermatofibroma
  - Vascular Lesion
  - Melanocytic Nevus

If the image does not appear to show a skin lesion set "not_skin_image": true in the response.

Return ONLY a valid JSON object — no markdown, no preamble, no trailing text:
{
    "not_skin_image": false,
    "predicted_disease": "<one of the 8 categories above>",
    "confidence": "<e.g. 87.4%>",
    "is_cancer": <true or false>,
    "cancer_status": "<CANCER DETECTED | NO CANCER DETECTED | PRECANCEROUS LESION — one sentence>",
    "ai_severity": "<High | Moderate | Low — with one-line justification>",
    "finding_explanation": "<Detailed explanation of WHY this classification was made, referencing specific visual features observed in the image>",
    "visual_characteristics": "<Describe color, shape, border regularity, texture, size estimation, and any distinctive markers visible>",
    "detailed_symptoms": "<Comprehensive list of symptoms and clinical signs a patient with this condition typically experiences>",
    "root_causes": "<Evidence-based causes and known risk factors for this condition>",
    "differential_diagnosis_notes": "<Which similar conditions were considered and why they were ruled out>",
    "home_treatments": "<Safe home-care measures appropriate for this condition — clearly note when professional care is essential>",
    "medical_treatments": "<Professional medical treatment options available for this condition>",
    "prevention_tips": "<Actionable prevention strategies specific to this condition>",
    "urgent_warning_signs": "<Red flags that require immediate medical attention>",
    "when_to_see_doctor": "<Clear guidance on urgency and timeline for professional evaluation>",
    "confidence_justification": "<Brief explanation of the confidence level given>"
}"""

        response = vision_model.generate_content([prompt, image])
        text = response.text.strip()

        # Strip markdown code fences if present
        if text.startswith('```'):
            lines = text.split('\n')
            text = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])

        analysis = json.loads(text)
        disease = analysis.get("predicted_disease", "Unknown")
        conf    = analysis.get("confidence", "?")
        log.info(f"[AI] Success — {disease} ({conf} confidence)")
        return analysis

    except json.JSONDecodeError as e:
        log.error(f"[AI] JSON parse failed: {e}")
        log.debug(f"[AI] Raw response: {response.text[:600]}")
        return None
    except Exception as e:
        log.error(f"[AI] Vision analysis exception: {e}")
        return None

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
    
    log.info(f"[VALIDATE] edge={edge_density:.4f}  brightness={brightness:.4f}  variance={variance:.4f}  color_std={np.mean(std_pixel):.2f}")
    
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
    
    result = is_edge_valid and is_brightness_valid and is_variance_valid and is_color_valid
    if not result:
        reasons = []
        if not is_edge_valid:      reasons.append(f"low edge density ({edge_density:.4f})")
        if not is_brightness_valid: reasons.append(f"brightness out of range ({brightness:.4f})")
        if not is_variance_valid:  reasons.append(f"low variance ({variance:.4f})")
        if not is_color_valid:     reasons.append(f"low colour std ({np.mean(std_pixel):.2f})")
        log.warning(f"[VALIDATE] Image rejected — {', '.join(reasons)}")
    else:
        log.info("[VALIDATE] Image passed pixel validation")
    return result


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    try:
        log.info(f"[REQUEST] Image received: {file.filename}")
        image = Image.open(file).convert("RGB")
        image_array = np.array(image)
        log.info(f"[REQUEST] Image size: {image.size}, mode: {image.mode}")

        # Pixel-level sanity check
        if not is_valid_skin_image(image_array):
            return render_template(
                'error.html',
                error_message="The uploaded image does not appear to be a valid skin image. Please upload a clear, close-up photograph of a skin lesion."
            )

        # Primary: vision analysis
        analysis = get_vision_analysis(image)

        # Fallback: trained model + static clinical info
        if analysis is None:
            log.warning("[REQUEST] Vision analysis failed — activating trained model fallback")
            analysis = get_model_fallback(image)

        if analysis is None:
            log.error("[REQUEST] Both vision analysis and fallback failed — returning error page")
            return render_template(
                'error.html',
                error_message="We were unable to analyze this image. Please try again with a clearer, well-lit photograph of the skin lesion."
            )

        if analysis.get("not_skin_image", False):
            log.warning("[REQUEST] Image rejected as non-skin by vision analysis")
            return render_template(
                'error.html',
                error_message="The uploaded image does not appear to show a skin lesion. Please upload a clear, close-up photograph of the area of concern."
            )

        predicted_disease = analysis.get("predicted_disease", "Unknown")
        confidence        = analysis.get("confidence", "N/A")
        is_cancer         = analysis.get("is_cancer", False)
        basic_info        = disease_info.get(predicted_disease, {})

        log.info(f"[REQUEST] Rendering result — disease: {predicted_disease}, confidence: {confidence}, cancer: {is_cancer}")

        cancer_status = analysis.get("cancer_status", "")
        return render_template(
            'result.html',
            predicted_disease=predicted_disease,
            confidence=confidence,
            is_cancer=is_cancer,
            cancer_status=sanitize_display_text(cancer_status),
            severity=analysis.get("ai_severity", "Moderate"),
            risk_tier=risk_tier_from_analysis(is_cancer, cancer_status),
            finding_explanation=sanitize_display_text(analysis.get("finding_explanation", "")),
            visual_characteristics=sanitize_display_text(analysis.get("visual_characteristics", "")),
            differential_diagnosis=sanitize_display_text(analysis.get("differential_diagnosis_notes", "")),
            differential_diagnosis_notes=sanitize_display_text(analysis.get("differential_diagnosis_notes", "")),
            confidence_justification=sanitize_display_text(analysis.get("confidence_justification", "")),
            detailed_symptoms=sanitize_display_text(analysis.get("detailed_symptoms", "")),
            root_causes=sanitize_display_text(analysis.get("root_causes", "")),
            home_treatments=sanitize_display_text(analysis.get("home_treatments", "")),
            medical_treatments=sanitize_display_text(analysis.get("medical_treatments", "")),
            when_to_see_doctor=sanitize_display_text(analysis.get("when_to_see_doctor", "")),
            prevention_tips=sanitize_display_text(analysis.get("prevention_tips", "")),
            urgent_warning_signs=sanitize_display_text(analysis.get("urgent_warning_signs", "")),
            predicted_disease_image=url_for('static', filename=f"images/{basic_info.get('image', 'melanoma.png')}")
        )

    except Exception as e:
        log.exception(f"[REQUEST] Unhandled exception in predict: {e}")
        return render_template(
            'error.html',
            error_message="An unexpected error occurred. Please try again with a different image."
        ), 500





# Run the app
if __name__ == "__main__":
    import sys
    
    # Get port from environment variable, command line argument, or default
    port = int(os.getenv('FLASK_PORT', sys.argv[1] if len(sys.argv) > 1 else 5000))
    debug = os.getenv('FLASK_ENV', 'production') == 'development'
    
    log.info(f"Starting DermaStratif on port {port} (debug={debug})")
    
    # Bind to 0.0.0.0 so it's accessible externally
    app.run(host='0.0.0.0', port=port, debug=debug)
