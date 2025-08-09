from flask import Flask, render_template, request, redirect, url_for, flash
import os
import uuid
import time
import logging
from werkzeug.utils import secure_filename
from transformers import pipeline
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/output'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Initialize the object detector with the same model as in zeroshot.py
logging.info("Loading zero-shot object detection model...")
checkpoint = "google/owlv2-base-patch16-ensemble"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection")
logging.info("Model loaded successfully!")

# Try to load a font, with fallback options
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=32)
except IOError:
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=32)
    except IOError:
        font = ImageFont.load_default()

# Define prices for each item
PRICES = {
    "Dairy Milk Snack Bar": 1.50,
    "Colgate Toothpaste": 3.25,
    "Cup Noodle Container": 0.99,
    "Croissant": 0.50,
    "Banana": 1.20,
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        logging.info("Received POST request for file upload")
        # Check if the post request has the file part
        if 'file' not in request.files:
            logging.warning("No file part in request")
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            logging.warning("No selected file")
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Generate a unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            logging.info(f"Saving uploaded file {file.filename} as {unique_filename}")
            # Save the uploaded file
            file.save(filepath)
            logging.info(f"File saved to {filepath}")
            
            # Process the image
            logging.info("Starting image processing")
            result_filename, detected_objects, total_price, processing_time = process_image(filepath, unique_filename)
            logging.info("Image processing completed")
            
            # Render the result template
            logging.info(f"Rendering result template with {len(detected_objects)} detected objects")
            return render_template('result.html', 
                                   image_file=result_filename, 
                                   detected_objects=detected_objects,
                                   total_price=total_price,
                                   processing_time=processing_time)
        else:
            logging.warning(f"Invalid file type: {file.filename}")
            flash('Invalid file type. Please upload a JPG, JPEG, or PNG file.')
            return redirect(request.url)
    
    return render_template('upload.html')

def process_image(filepath, filename):
    logging.info(f"Processing image: {filename}")
    start_time = time.time()
    
    # Open the image
    with Image.open(filepath) as image:
        # Process image the same way as in zeroshot.py
        image = Image.fromarray(np.uint8(image)).convert("RGB")
        logging.info(f"Image loaded and converted to RGB")
        
        # Run object detection with the same candidate labels as in zeroshot.py
        logging.info(f"Running object detection with labels: {list(PRICES.keys())}")
        detection_start = time.time()
        predictions = detector(
            image,
            candidate_labels=list(PRICES.keys()),
        )
        detection_time = time.time() - detection_start
        logging.info(f"Detection completed in {detection_time:.2f} seconds")
        logging.info(f"Raw predictions: {predictions}")
        
        draw = ImageDraw.Draw(image)
        
        final_pred = defaultdict(list)
        
        # Define confidence threshold
        CONFIDENCE_THRESHOLD = 0.1
        logging.info(f"Using confidence threshold: {CONFIDENCE_THRESHOLD}")
        
        # Process predictions using the same logic as in zeroshot.py
        filtered_count = 0
        for prediction in predictions:
            box = prediction["box"]
            label = prediction["label"]
            score = prediction["score"]
            
            # Skip predictions with confidence below threshold
            if score < CONFIDENCE_THRESHOLD:
                logging.info(f"Filtering out {label} with low confidence: {score:.4f}")
                continue
            else:
                logging.info(f"Keeping prediction: {label} with confidence: {score:.4f}")
                filtered_count += 1

            xmin, ymin, xmax, ymax = box.values()
            size = (xmax - xmin) * (ymax - ymin)
            
            if not final_pred[label]:
                final_pred[label] = [size, xmin, ymin, xmax, ymax, score]
                logging.info(f"Added first detection for {label}")
            else:
                if final_pred[label][5] < score:
                    logging.info(f"Replaced detection for {label} with higher confidence: {score:.4f} > {final_pred[label][5]:.4f}")
                    final_pred[label] = [size, xmin, ymin, xmax, ymax, score]
        
        logging.info(f"Kept {filtered_count} predictions after confidence filtering")
        
        # Create detected_objects dictionary with count=1 for each detected item
        # This ensures we only count each unique item once and applies the confidence threshold
        detected_objects = {}
        for label, values in final_pred.items():
            # Only include items that meet the confidence threshold
            if values[5] >= CONFIDENCE_THRESHOLD:
                detected_objects[label] = 1
                logging.info(f"Including {label} in final results with confidence {values[5]:.4f}")
        
        logging.info(f"Final detected objects: {detected_objects}")
        
        # Draw bounding boxes and labels only for items that meet the confidence threshold
        logging.info("Drawing bounding boxes on image")
        for label, values in final_pred.items():
            _, xmin, ymin, xmax, ymax, score = values
            # Only draw if the score meets the threshold
            if score >= CONFIDENCE_THRESHOLD:
                draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=5)
                draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="white", font=font)
                logging.info(f"Drew bounding box for {label} at ({xmin}, {ymin}, {xmax}, {ymax})")
        
        # Save the processed image
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], os.path.basename(filename))
        image.save(output_path)
        result_filename = os.path.basename(filename)
        logging.info(f"Saved processed image to {output_path}")
        
        # Calculate total price
        total_price = sum(PRICES[item] * count for item, count in detected_objects.items())
        logging.info(f"Calculated total price: ${total_price:.2f}")
        
        # Log total processing time
        total_time = time.time() - start_time
        logging.info(f"Total processing time: {total_time:.2f} seconds")
        
        return result_filename, detected_objects, total_price, total_time

if __name__ == '__main__':
    app.run(debug=True)
