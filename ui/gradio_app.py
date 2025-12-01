"""
Gradio web interface for the sentiment analysis API.
"""
import gradio as gr
import requests
import pandas as pd
from typing import Tuple, Optional
from datetime import datetime

# API endpoints
API_BASE = "http://localhost:8000"

def predict_sentiment(text: str) -> Tuple[str, float, str, str]:
    """
    Call the prediction API and return results.
    
    Returns:
        (label, confidence, confidence_bar, status_message)
    """
    if not text.strip():
        return "N/A", 0.0, "", "Please enter some text"
    
    try:
        response = requests.post(
            f"{API_BASE}/v1/predict",
            json={"text": text},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            label = data['label']
            confidence = data['confidence']
            
            # Create confidence bar
            conf_pct = int(confidence * 100)
            
            # Determine emoji and color
            if confidence >= 0.8:
                emoji = "✅"
                status = f"{emoji} High confidence ({conf_pct}%)"
            elif confidence >= 0.55:
                emoji = "🟡"
                status = f"{emoji} Medium confidence ({conf_pct}%)"
            else:
                emoji = "⚠️"
                status = f"{emoji} Low confidence ({conf_pct}%) - Logged for review"
            
            return label.upper(), confidence, f"{conf_pct}%", status
        else:
            return "ERROR", 0.0, "", f"❌ API Error: {response.status_code}"
    
    except Exception as e:
        return "ERROR", 0.0, "", f"❌ Connection Error: {str(e)}"

def submit_correction(
    text: str, 
    predicted_label: str, 
    confidence: float,
    correct_label: str
) -> str:
    """
    Submit feedback to the API.
    
    Returns:
        Status message
    """
    if not text.strip():
        return "No prediction to correct"
    
    if predicted_label == "N/A" or predicted_label == "ERROR":
        return "Please make a prediction first"
    
    if not correct_label or correct_label == "Choose...":
        return "Please select the correct label"
    
    try:
        response = requests.post(
            f"{API_BASE}/v1/feedback",
            json={
                "text": text,
                "predicted_label": predicted_label.lower(),
                "correct_label": correct_label,
                "confidence": confidence
            },
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            return f"{data['message']} (ID: {data['feedback_id']})"
        else:
            return f"Failed to submit feedback: {response.status_code}"
    
    except Exception as e:
        return f"Error: {str(e)}"

def get_model_stats() -> str:
    """Fetch current model statistics."""
    try:
        response = requests.get(f"{API_BASE}/v1/model_info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return f"""
### Current Model Stats
- **Version:** {data['version']}
- **Accuracy:** {data['accuracy']:.2%}
- **Total Predictions:** {data['total_predictions']}
- **Total Feedback:** {data['total_feedback']}
            """
        else:
            return "Failed to fetch stats"
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Sentiment Analysis", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Sentiment Analysis Self-Learning System")
    gr.Markdown("Analyze text sentiment and help improve the model by correcting mistakes.")
    
    with gr.Tab("Analyze"):
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="Enter text to analyze",
                    placeholder="Type or paste your text here...",
                    lines=5
                )
                analyze_btn = gr.Button("Analyze Sentiment", variant="primary", size="lg")
                
                # Example texts
                gr.Examples(
                    examples=[
                        "I absolutely loved this movie! Best film I've seen all year.",
                        "Terrible experience, worst customer service ever.",
                        "It's okay, nothing special but not bad either.",
                        "The product works fine but arrived damaged.",
                    ],
                    inputs=input_text,
                    label="Try these examples"
                )
            
            with gr.Column():
                predicted_label_display = gr.Textbox(
                    label="Predicted Sentiment",
                    interactive=False,
                    scale=1
                )
                confidence_display = gr.Textbox(
                    label="Confidence Score",
                    interactive=False,
                    scale=1
                )
                status_display = gr.Markdown("Ready to analyze")
        
        # Store prediction state (hidden)
        predicted_label_hidden = gr.State()
        confidence_hidden = gr.State()
        
        gr.Markdown("---")
        gr.Markdown("### Was this prediction wrong?")
        gr.Markdown("Help improve the model by submitting a correction:")
        
        with gr.Row():
            correct_label = gr.Dropdown(
                choices=["positive", "negative", "neutral"],
                label="What should the correct label be?",
                scale=2
            )
            submit_btn = gr.Button("Submit Correction", variant="secondary", scale=1)
        
        correction_status = gr.Markdown()
        
        # Wire up the prediction
        analyze_btn.click(
            fn=predict_sentiment,
            inputs=[input_text],
            outputs=[
                predicted_label_display,
                confidence_hidden,
                confidence_display,
                status_display
            ]
        ).then(
            fn=lambda x: x.lower() if x not in ["N/A", "ERROR"] else x,
            inputs=[predicted_label_display],
            outputs=[predicted_label_hidden]
        )
        
        # Wire up the correction
        submit_btn.click(
            fn=submit_correction,
            inputs=[input_text, predicted_label_hidden, confidence_hidden, correct_label],
            outputs=[correction_status]
        )
    
    with gr.Tab("Model Info"):
        gr.Markdown("## Current Model Information")
        stats_display = gr.Markdown()
        refresh_btn = gr.Button("Refresh Stats")
        
        refresh_btn.click(
            fn=get_model_stats,
            outputs=[stats_display]
        )
        
        # Load stats on initial render
        demo.load(fn=get_model_stats, outputs=[stats_display])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
