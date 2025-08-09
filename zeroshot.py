from transformers import pipeline
import skimage
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import sys


from collections import defaultdict

checkpoint = "google/owlv2-base-patch16-ensemble"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection")

font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=32)

for idx in range(1, 8):
    print(idx)
    with Image.open(f"inp/example{idx}.jpg") as image:    

        image = Image.fromarray(np.uint8(image)).convert("RGB")

        predictions = detector(
            image,
            candidate_labels=["Dairy Milk Snack Bar", "Colgate Toothpaste", "Cup Noodle Container"],
        )

        print(predictions)

        draw = ImageDraw.Draw(image)

        final_pred = defaultdict(list)

        for prediction in predictions:
            box = prediction["box"]
            label = prediction["label"]
            score = prediction["score"]

            xmin, ymin, xmax, ymax = box.values()
            size = (xmax - xmin) * (ymax - ymin)
            
            if not final_pred[label]:
                final_pred[label] = [size, xmin, ymin, xmax, ymax, score]
            else:
                if final_pred[label][5] < score:
                    final_pred[label] = [size, xmin, ymin, xmax, ymax, score]
                    
        print(final_pred)
        
        for label, values in final_pred.items():
            _, xmin, ymin, xmax, ymax, score = values
            draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=5)
            draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="white", font=font)
            
        image.save(f"./output/res{idx}.png", "PNG")