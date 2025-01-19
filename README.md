# DermaStratif - Multiclass Lesion Stratification and Diagnosis

This repository implements EfficientNet-B0 for skin lesion classification using three different fine-tuning methods:

1. Base Model Training
2. Adapter Fine-Tuning
3. LoRA Fine-Tuning

The objective is to classify skin lesion images into 8 disease categories while improving accuracy and efficiency using advanced fine-tuning techniques.

---

## Project Overview

### Base Model
- **Model**: EfficientNet-B0 pre-trained on ImageNet.
- **Classifier**: Fully connected layers added to support 8 disease categories.
- **Validation Accuracy**: 74%
- **Training Time**: ~4 hours on a MacBook Pro M1 GPU.
- **Description**: The base model was trained with all layers unfrozen to fine-tune for skin lesion classification. It serves as the foundation for further improvements using Adapter and LoRA fine-tuning.

### Adapter Fine-Tuning
- **What Are Adapters?**
  - Small, lightweight layers added to the model's classifier for task-specific fine-tuning.
  - Parameters outside these adapters remain frozen, reducing computational cost.
- **Validation Accuracy**: 76%
- **Training Time**: ~1 hour.
- **Description**: Adapter fine-tuning improves resource efficiency while achieving better accuracy than the base model.

### LoRA Fine-Tuning
- **What Is LoRA?**
  - Low-Rank Adaptation (LoRA) introduces task-specific low-rank matrices into the model's classifier.
  - Significantly reduces memory usage while maintaining high accuracy.
- **Validation Accuracy**: 80%
- **Training Time**: ~45 minutes.
- **LoRA Configuration**:
  - Rank (`r`): 8
  - Scaling Factor (`alpha`): 32
- **Description**: LoRA fine-tuning yielded the highest validation accuracy with minimal computational cost.

---

## Disease Classes

The model classifies skin lesions into the following 8 categories:

1. **Melanoma**: High-risk skin cancer with irregular moles.
2. **Basal Cell Carcinoma**: Pearly or waxy bumps on the skin.
3. **Squamous Cell Carcinoma**: Firm, red nodules with scaly, crusted surfaces.
4. **Actinic Keratosis**: Rough, scaly patches commonly found on sun-exposed skin.
5. **Benign Keratosis**: Non-cancerous skin growths appearing as brown or black patches.
6. **Dermatofibroma**: Small, firm nodules caused by minor injuries like insect bites.
7. **Vascular Lesion**: Marks or growths caused by abnormal blood vessels.
8. **Melanocytic Nevus**: Common moles that are small and pigmented.

---

## Application Details

### How to Use

1. **Download the Repository**:
   Clone or download the repository files:
   ```bash
   git clone <repository-url>
2. **Prepare Required Files:**
- Download the application folder and the saved model folder.
- Update the saved_model_path in app.py to the path of your downloaded model.
3. **Install Dependencies**: Install the required Python libraries:
- pip install -r requirements.txt
4. **Run the Application**: Start the Flask server by running:
- <b>python app.py</b>
5. **Access the Application**: Open your browser and navigate to:
- http://127.0.0.1:5000/
6. **Upload an Image**: Use the upload form to submit a skin lesion image and view the predictions.


### Dependencies
Install all dependencies using the following command:

- pip install -r requirements.txt
#### Key Libraries:
- Flask
- PyTorch
- torchvision
- Pillow
- scikit-learn

### Future Improvements
- Support for larger models (e.g., EfficientNet-B3 or ViT) for higher accuracy.
- Provide a live API for disease prediction.
- Enhance the UI with interactive features and visualization tools.
