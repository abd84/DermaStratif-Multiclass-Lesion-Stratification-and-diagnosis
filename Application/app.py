import os
import logging
import json
from flask import Flask, request, render_template, url_for, jsonify
from PIL import Image
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
    log.warning("Vision API key not found — analysis will be unavailable")

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


def get_vision_analysis(image):
    """Analyze the skin lesion image via Gemini and return a clinical assessment dict, or None on failure."""
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
        return None
    except Exception as e:
        log.error(f"[AI] Vision analysis exception: {e}")
        return None


# Disease information for static image mapping
disease_info = {
    "Melanoma":                {"image": "melanoma.png"},
    "Melanocytic Nevus":       {"image": "melanocytic_nevus.png"},
    "Basal Cell Carcinoma":    {"image": "basal_cell_carcinoma.png"},
    "Actinic Keratosis":       {"image": "actinic_keratosis.png"},
    "Benign Keratosis":        {"image": "benign_keratosis.png"},
    "Dermatofibroma":          {"image": "dermatofibroma.png"},
    "Vascular Lesion":         {"image": "vascular_lesion.png"},
    "Squamous Cell Carcinoma": {"image": "squamous_cell_carcinoma.png"},
}


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


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
        log.info(f"[REQUEST] Image size: {image.size}")

        analysis = get_vision_analysis(image)

        if analysis is None:
            log.error("[REQUEST] Vision analysis failed — returning error page")
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
        cancer_status     = analysis.get("cancer_status", "")
        basic_info        = disease_info.get(predicted_disease, {})

        log.info(f"[REQUEST] Rendering result — disease: {predicted_disease}, confidence: {confidence}, cancer: {is_cancer}")

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


if __name__ == "__main__":
    import sys
    port = int(os.getenv('FLASK_PORT', sys.argv[1] if len(sys.argv) > 1 else 5000))
    debug = os.getenv('FLASK_ENV', 'production') == 'development'
    log.info(f"Starting DermaStratif on port {port} (debug={debug})")
    app.run(host='0.0.0.0', port=port, debug=debug)
