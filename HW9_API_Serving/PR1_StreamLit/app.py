import streamlit as st
import requests
import json
import numpy as np
import pandas as pd
from PIL import Image
import io

# Configuration
API_URL = "http://localhost:8000/predict"

# Set page title and favicon
st.set_page_config(
    page_title="Agricultural Threat Detector",
    page_icon="🌱",
    layout="wide"
)

# Header
st.title("Agricultural Threat Detection System")
st.markdown("Upload an image to identify potential threats to your crops")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.info("This application uses a machine learning model to detect diseases, pests, and weeds that threaten agricultural crops.")
    
    st.header("How to use")
    st.write("1. Upload an image of the plant or crop")
    st.write("2. Adjust confidence threshold if needed")
    st.write("3. Click 'Analyze Image'")
    st.write("4. View results and recommendations")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Confidence threshold
confidence = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)

# Process the image
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert image for API
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format)
    img_byte_arr = img_byte_arr.getvalue()
    
    # Create predict button
    if st.button("Analyze Image"):
        with st.spinner("Analyzing..."):
            try:
                # Call the API
                files = {"file": img_byte_arr}
                params = {"confidence": confidence}
                
                response = requests.post(
                    API_URL, 
                    files=files,
                    params=params
                )
                
                if response.status_code == 200:
                    # Extract prediction results
                    results = response.json()
                    
                    # Display results
                    st.subheader("Analysis Results")
                    
                    # Create two columns for results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Detected Threats")
                        if len(results["threats"]) > 0:
                            threat_data = []
                            for threat in results["threats"]:
                                threat_data.append({
                                    "Threat Type": threat["type"],
                                    "Confidence": f"{threat['confidence']:.2%}",
                                    "Name": threat["name"]
                                })
                            st.table(pd.DataFrame(threat_data))
                        else:
                            st.success("No threats detected!")
                    
                    with col2:
                        st.markdown("### Recommendations")
                        if len(results["recommendations"]) > 0:
                            for rec in results["recommendations"]:
                                st.info(rec)
                        else:
                            st.info("No specific recommendations available.")
                    
                    # Additional information if available
                    if "details" in results and results["details"]:
                        with st.expander("Additional Details"):
                            st.json(results["details"])
                
                else:
                    st.error(f"Error: {response.status_code}, {response.text}")
            
            except Exception as e:
                st.error(f"Error connecting to the API: {str(e)}")

# Footer
st.markdown("---")
st.markdown("© 2023 Agricultural Threat Detection System | v1.0.0")
