from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import json
import os

app = FastAPI(title="Cloth Type Classification API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model variables
model = None
idx_to_name = None
not_clothing_idx = None
clothing_classes = None

# Preprocessing - EXACTLY from your app
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.on_event("startup")
async def load_model():
    global model, idx_to_name, not_clothing_idx, clothing_classes
    
    try:
        # Load model checkpoint - EXACTLY like your app
        MODEL_PATH = "laundry_sorter_robust.pth"
        MAPPING_PATH = "final_class_mapping.json"
        
        # Load class mapping
        with open(MAPPING_PATH, 'r') as f:
            mapping_data = json.load(f)
        idx_to_name = mapping_data["idx_to_name"]
        
        # Find "Not Clothing" index - EXACTLY like your app
        not_clothing_idx = next(i for i, n in enumerate(idx_to_name) if "Not Clothing" in n)

        # Load model - EXACTLY like your app
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        
        model = models.resnet18(pretrained=False)
        model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, len(idx_to_name)))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Get clothing classes - EXACTLY like your app
        clothing_classes = [
            name for i, name in enumerate(idx_to_name)
            if i != not_clothing_idx
        ]
        
        print("✅ Cloth Type Model loaded successfully!")
        print(f"📊 Available classes: {clothing_classes}")
        print(f"🚫 Not Clothing index: {not_clothing_idx}")
        
    except Exception as e:
        print(f"❌ Error loading cloth type model: {e}")
        raise e

@app.get("/")
async def root():
    return {"message": "Cloth Type Classification API is running!"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model else "error",
        "model_loaded": model is not None,
        "available_classes": clothing_classes,
        "not_clothing_index": not_clothing_idx
    }

@app.get("/classes")
async def get_classes():
    """Get all available clothing classes"""
    return {
        "clothing_classes": clothing_classes,
        "not_clothing_class": "Not Clothing",
        "all_classes": idx_to_name
    }

@app.post("/predict")
async def predict_cloth_type(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and validate image
        image_data = await file.read()
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
            
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Preprocess - EXACTLY like your app
        img_tensor = preprocess(image).unsqueeze(0)
        
        # Predict - EXACTLY like your app
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]
        
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()
        
        # Apply confidence threshold for "Not Clothing" - EXACTLY like your app
        predicted_class = idx_to_name[pred_idx]
        final_class = predicted_class.split(",")[0]  # Take first part before comma
        
        if pred_idx != not_clothing_idx and confidence < 0.60:
            final_class = "Not Clothing"
            confidence = 1.0 - confidence  # Invert confidence for rejection
        
        # Prepare all predictions
        all_predictions = {}
        for i, class_name in enumerate(idx_to_name):
            simple_name = class_name.split(",")[0]
            all_predictions[simple_name] = float(probs[i])
        
        return {
            "cloth_type": final_class,
            "confidence": float(confidence),
            "original_prediction": predicted_class,
            "all_predictions": all_predictions,
            "is_clothing": final_class != "Not Clothing"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)