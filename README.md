# Zero-Shot Object Detection Demo

This is a web application that demonstrates zero-shot object detection using the OwlV2 model from Google. The application allows users to upload images or capture them using a webcam, detect specific objects, calculate their prices, and display the results.

## Features

- Upload images for object detection
- Capture images directly from webcam
- Detect predefined items using zero-shot object detection
- Draw bounding boxes around detected objects
- Calculate total price of detected items
- Filter low-confidence detections
- Visual progress indicator during processing
- Processing time display
- Model caching for faster inference

## Objects and Prices

The application can detect the following items:
- Dairy Milk Snack Bar: $1.50
- Colgate Toothpaste: $3.25
- Cup Noodle Container: $0.99
- Croissant: $0.50
- Banana: $1.20

## Installation

Clone the repository into your system:
```
git clone https://github.com/dthxe/obj-detection-app
```

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python app.py
   ```

3. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

## Usage

1. Click the "Choose File" button to select an image from your computer
2. Click "Upload and Detect" to process the image
3. View the results showing detected objects, quantities, and prices
4. Click "Upload Another Image" to try with a different image

## Technical Details

This application uses:
- Flask for the web interface
- Hugging Face Transformers library with the OwlV2 model for zero-shot object detection
- PIL for image processing and drawing bounding boxes
- Bootstrap for styling

## Directory Structure

- `app.py`: Main application file
- `templates/`: HTML templates
- `uploads/`: Temporary storage for uploaded images
- `static/output/`: Processed images with bounding boxes
