import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
import timm
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Chest X-Ray Disease Classifier",
    layout="wide"
)

# Disease classes and optimal thresholds
DISEASE_CLASSES = [
    'Atelectasis', 'Effusion', 'Infiltration', 'Mass', 
    'No Finding', 'Nodule', 'Pneumothorax'
]

OPTIMAL_THRESHOLDS = {
    'Atelectasis': 0.35,
    'Effusion': 0.35,
    'Infiltration': 0.30,
    'Mass': 0.35,
    'No Finding': 0.45,
    'Nodule': 0.20,
    'Pneumothorax': 0.50
}

# Load model
@st.cache_resource
def load_model():
    """Load the trained ResNet50 model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = timm.create_model('resnet50', pretrained=False, num_classes=len(DISEASE_CLASSES))
    
    # Load weights
    try:
        model.load_state_dict(torch.load('best_resnet50.pth', map_location=device))
        model.eval()
        model.to(device)
        return model, device
    except FileNotFoundError:
        st.error("Error: Model file 'best_resnet50.pth' not found. Please ensure it is in the same directory as app.py")
        return None, device

# Image preprocessing
def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict_image(image, model, device):
    """Make predictions on uploaded image"""
    # Preprocess
    img_tensor = preprocess_image(image).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
    
    # Apply optimal thresholds
    predictions = []
    for i, disease in enumerate(DISEASE_CLASSES):
        prob = probabilities[i]
        threshold = OPTIMAL_THRESHOLDS[disease]
        if prob > threshold:
            predictions.append({
                'disease': disease,
                'probability': prob,
                'threshold': threshold
            })
    
    # Sort by probability
    predictions.sort(key=lambda x: x['probability'], reverse=True)
    
    return predictions, probabilities

# Main app
def main():
    # Header
    st.title("Chest X-Ray Disease Classifier")
    st.markdown("""
    Multi-label classification of chest diseases using ResNet50 deep learning model.  
    Upload a chest X-ray image to detect 7 common pathologies.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("Model Information")
        st.markdown("""
        **Architecture:** ResNet50  
        **Parameters:** 23.5M  
        **Dataset:** NIH Chest X-Ray  
        **Performance:** F1 0.442  
        """)
        
        st.markdown("---")
        
        st.header("Disease Classes")
        for disease in DISEASE_CLASSES:
            st.text(f"• {disease}")
        
        st.markdown("---")
        
        st.caption("Disclaimer: For research and educational purposes only. Not intended for clinical use.")
    
    # Load model
    model, device = load_model()
    
    if model is None:
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Chest X-Ray Image",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Predictions")
            
            # Predict
            with st.spinner('Analyzing...'):
                predictions, all_probs = predict_image(image, model, device)
            
            # Display predictions
            if predictions:
                st.markdown("**Detected Pathologies:**")
                for pred in predictions:
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(f"**{pred['disease']}**")
                    with col_b:
                        st.markdown(f"`{pred['probability']*100:.1f}%`")
                    
                    # Progress bar (convert numpy float to Python float)
                    st.progress(float(pred['probability']))
                    st.caption(f"Threshold: {pred['threshold']:.2f}")
                    st.markdown("---")
            else:
                st.info("No significant findings detected. All probabilities below threshold.")
        
        # Probability visualization
        st.markdown("---")
        st.subheader("Disease Probability Distribution")
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color bars based on threshold
        colors = ['#ff4444' if all_probs[i] > OPTIMAL_THRESHOLDS[disease] else '#cccccc' 
                  for i, disease in enumerate(DISEASE_CLASSES)]
        
        bars = ax.barh(DISEASE_CLASSES, all_probs, color=colors)
        
        # Add threshold lines
        for i, disease in enumerate(DISEASE_CLASSES):
            thresh = OPTIMAL_THRESHOLDS[disease]
            ax.plot([thresh, thresh], [i-0.4, i+0.4], 'k--', linewidth=2, alpha=0.5)
        
        ax.set_xlabel('Probability', fontsize=11)
        ax.set_title('Disease Probabilities with Optimal Thresholds', fontsize=13, pad=15)
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add probability labels
        for i, (disease, prob) in enumerate(zip(DISEASE_CLASSES, all_probs)):
            ax.text(prob + 0.02, i, f'{prob:.3f}', va='center', fontsize=10)
        
        st.pyplot(fig)
        
        # Detailed metrics
        with st.expander("Detailed Metrics"):
            st.markdown("### Per-Disease Analysis")
            
            for i, disease in enumerate(DISEASE_CLASSES):
                prob = all_probs[i]
                thresh = OPTIMAL_THRESHOLDS[disease]
                status = "DETECTED" if prob > thresh else "Not Detected"
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(disease, f"{prob*100:.1f}%", delta=f"{((prob-thresh)/thresh*100):+.1f}%" if prob > thresh else None)
                with col2:
                    st.caption(f"Threshold: {thresh:.2f}")
                with col3:
                    st.caption(f"Status: {status}")
                
                st.markdown("---")

if __name__ == "__main__":
    main()
