# Zero-Shot Object Detection Web Demo

This project performs object detection using a transformer-based model** with an interactive web interface.  
It explores the zero-shot object detection capabilities of models like Google's OwlV2 to identify objects from arbitrary text prompts without retraining. It also explores the ability to improve the model for specific applications via fine-tuning on custom datasets.  
The application is built with a modular architecture that allows easy extension to new detection targets, UI components, and backend services. 

---

## Key Technical Features

- **Zero-Shot Detection and Transformer Fine-tuning Pipeline**  
  Utilizes Hugging Face's implementation of the OwlV2 model for object detection without task-specific training.
  
- **Web-Based Interaction**  
  Flask-powered backend serving a responsive interface for image submission and result visualization.
  
- **Multi-Source Input Handling**  
  Supports both file uploads and live webcam capture directly from the browser.
  
- **On-the-Fly Image Annotation**  
  Uses Pillow (PIL) to preprocess images, draw bounding boxes, and render annotated outputs.
  
- **Confidence Filtering**  
  Dynamically filters low-confidence detections to improve result quality.
  
- **Performance Optimizations**  
  Caches loaded models to minimize inference time and supports progressive status updates during processing.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/dthxe/obj-detection-app
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Access via browser:

```
http://127.0.0.1:5000
```

---

## Usage

1. **Launch** the application locally.  
2. **Provide an image** via upload or webcam capture.  
3. **Initiate detection** – the system will identify objects matching predefined or custom text queries.  
4. **View annotated results** in the browser.  
5. **Iterate** with different images or class lists without restarting.

---

## Technology Stack

- **Backend Framework:** Flask (Python)  
- **Model Inference:** Hugging Face Transformers (OwlV2)  
- **Image Processing:** Pillow (PIL)  
- **Frontend Styling:** Bootstrap  
- **Webcam Integration:** HTML5 Media Capture APIs

---

## Directory Structure

```
app.py               # Main Flask application logic
templates/           # HTML templates (Jinja2)
uploads/             # Temporary storage for raw input images
static/output/       # Model-annotated images for display
```

---

## Project Notes

This repository serves as a technical reference implementation for deploying transformer-based zero-shot object detection in a browser-accessible environment.  
The modular design allows for quick adaptation to new domains — changing the target label set or extending the preprocessing pipeline requires minimal code changes.
