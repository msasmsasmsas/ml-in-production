import gradio as gr
import requests
import numpy as np
import pandas as pd
import json
from PIL import Image
import io

# Configuration
API_URL = "http://localhost:8000/predict"

def predict_image(image, confidence):
    """
    Send image to the API and get prediction results
    
    Args:
        image: PIL Image object
        confidence: float threshold value
        
    Returns:
        tuple: (detected threats table, recommendations, details JSON)
    """
    # Convert image for API
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)  # Важно: вернуть указатель в начало потока


    try:
        # Call the API
        # files = {"file": ("image.png", img_byte_arr, "image/png")}
        files = {"file": ("image.png", img_byte_arr, "image/png")}
        params = {"confidence": confidence}
        
        response = requests.post(API_URL, files=files, params=params)
        
        if response.status_code == 200:
            results = response.json()
            
            # Process threats data for display
            if results["threats"]:
                threats_df = pd.DataFrame(results["threats"])
                # Format confidence as percentage
                threats_df["confidence"] = threats_df["confidence"].apply(lambda x: f"{x:.2%}")
                threats_table = threats_df[["type", "name", "confidence"]].to_markdown()
            else:
                threats_table = "No threats detected in this image."
            
            # Join recommendations
            recommendations = "\n".join(results["recommendations"]) if results["recommendations"] else "No specific recommendations available."
            
            # Format details as pretty JSON
            details = json.dumps(results.get("details", {}), indent=2) if "details" in results else ""
            
            return threats_table, recommendations, details
        else:
            return f"Error: {response.status_code}", f"API Error: {response.text}", ""
            
    except Exception as e:
        return f"Error connecting to the API", f"Details: {str(e)}", ""

# Define the Gradio interface
with gr.Blocks(title="Agricultural Threat Detector", theme=gr.themes.Soft()) as app:
    gr.Markdown("# Agricultural Threat Detection System")
    gr.Markdown("Upload an image to identify potential threats to your crops")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input components
            input_image = gr.Image(type="pil", label="Upload Image")
            confidence = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.5, 
                                  label="Confidence Threshold")
            submit_button = gr.Button("Analyze Image", variant="primary")
            
        with gr.Column(scale=1):
            # Output components
            threats_output = gr.Markdown(label="Detected Threats")
            recommendations_output = gr.Textbox(label="Recommendations", lines=5)
            details_output = gr.Code(language="json", label="Additional Details")
    
    # Event handler
    submit_button.click(
        fn=predict_image,
        inputs=[input_image, confidence],
        outputs=[threats_output, recommendations_output, details_output]
    )
    
    # Example images
    gr.Examples(
        examples=[
            ["example_images/healthy_wheat.jpg", 0.5],
            ["example_images/wheat_rust.jpg", 0.5],
            ["example_images/corn_blight.jpg", 0.5],
        ],
        inputs=[input_image, confidence],
    )
    
    # Information tab
    with gr.Accordion("How to use", open=False):
        gr.Markdown("""
        ## Instructions
        1. Upload an image of the plant or crop you want to analyze
        2. Adjust the confidence threshold as needed (higher values mean higher confidence required)
        3. Click 'Analyze Image'
        4. View the results including detected threats, recommendations, and additional details
        
        ## About
        This application uses a machine learning model to detect diseases, pests, and weeds 
        that threaten agricultural crops. The model analyzes the visual patterns in the image 
        to identify potential issues and provide relevant recommendations.
        """)

# Launch the app
if __name__ == "__main__":
    app.launch(share=False)
