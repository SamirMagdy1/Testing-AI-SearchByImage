from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import io

app = FastAPI()

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

@app.post("/classify-image/")
async def classify_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    predicted_class_label = model.config.id2label[predicted_class_idx]

    return JSONResponse(content={"predicted_class": predicted_class_label})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
