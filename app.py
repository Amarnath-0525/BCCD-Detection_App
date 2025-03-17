import gradio as gr
from ultralytics import YOLO
from PIL import Image
import os

model_path = "best.pt"

if os.path.exists(model_path):
    model = YOLO(model_path) 
    print("✅ Model loaded successfully!")
else:
    raise FileNotFoundError(f"❌ Model file not found at {model_path}")

def detect_objects(image):

    results = model(image)  

    output_image_path = "output.jpg"
    results[0].save(output_image_path)

    return Image.open(output_image_path)

interface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil"),  
    outputs=gr.Image(type="pil"),  
    title="BCCD YOLO Object Detection",
    description="Upload a blood cell image to detect RBCs, WBCs, and Platelets."
)

interface.launch(share=True)
