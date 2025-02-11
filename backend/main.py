import os
import shutil
import torch
import uvicorn
import gc
import datetime
import numpy as np
import subprocess
import pandas as pd
from tqdm import tqdm
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from icrawler.builtin import GoogleImageCrawler
from PIL import Image
from transformers import pipeline
from diffusers import DiffusionPipeline, AutoencoderKL
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount the directory where generated images are stored
app.mount("/images", StaticFiles(directory=".", html=True), name="images")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change to ["http://localhost:5173"] for more security
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI"}

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Paths
IMAGE_FOLDER = "./images"
RESIZED_FOLDER = "./resized_images"
FACES_FOLDER = "./faces"
DATA_FOLDER = "./dataset"

# Ensure directories exist
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(RESIZED_FOLDER, exist_ok=True)
os.makedirs(FACES_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# Initialize BLIP-2 captioning model
captioner = pipeline("image-to-text", model="Salesforce/blip2-opt-2.7b", device=device)

# Load LoRA model with memory optimization
try:
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae, torch_dtype=torch.float16, variant="fp16"
    )
    
    # Enable memory-efficient attention
    pipeline.enable_xformers_memory_efficient_attention()
    
    pipeline.to(device)
    print("✅ LoRA model loaded successfully with memory optimizations")
except Exception as e:
    print(f"❌ Failed to load LoRA model: {e}")
    pipeline = None


# IMAGE SCRAPING
import random

@app.get("/scrape_images/")
async def scrape_images(keyword: str, max_num: int = 60):
    """Scrapes images, then randomly selects 10 to keep."""
    google_crawler = GoogleImageCrawler(storage={'root_dir': IMAGE_FOLDER})
    google_crawler.crawl(keyword=keyword, max_num=max_num)

    # List all downloaded images
    images = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith((".png", ".jpg", ".jpeg"))]
    
    if len(images) > 10:
        # Randomly select 10 images
        selected_images = random.sample(images, 10)

        # Delete the rest
        for img in images:
            if img not in selected_images:
                os.remove(os.path.join(IMAGE_FOLDER, img))
    
    return {"message": f"✅ Scraped {max_num} images, randomly kept 10 for '{keyword}'"}


# IMAGE UPLOAD
@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    """Uploads an image."""
    file_path = os.path.join(IMAGE_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"message": "✅ Image uploaded successfully", "filename": file.filename}


# IMAGE PROCESSING (Resizing)
@app.post("/process_images/")
async def process_images():
    """Resizes images to 1024x1024 and prepares dataset."""
    if os.path.exists(RESIZED_FOLDER):
        shutil.rmtree(RESIZED_FOLDER)
    os.makedirs(RESIZED_FOLDER, exist_ok=True)

    # Only process 10 images
    images = sorted(os.listdir(IMAGE_FOLDER))[:10]  # Select first 10 images

    for file_name in images:
        file_path = os.path.join(IMAGE_FOLDER, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                img = Image.open(file_path).convert("RGB")
                img = img.resize((1024, 1024))
                img.save(os.path.join(RESIZED_FOLDER, file_name))
            except Exception as e:
                print(f"❌ Error processing {file_name}: {e}")

    return {"message": "✅ Images processed and resized"}


# IMAGE CAPTIONING WITH BLIP-2
@app.post("/generate_captions/")
async def generate_captions():
    """Generates captions for images using BLIP-2."""
    captions = {}

    for file_name in tqdm(os.listdir(RESIZED_FOLDER)):
        file_path = os.path.join(RESIZED_FOLDER, file_name)
        img = Image.open(file_path)
        caption = captioner(img)[0]['generated_text'].strip()
        captions[file_name] = caption
        img.save(os.path.join(FACES_FOLDER, file_name))

    # Save captions to CSV
    df = pd.DataFrame.from_dict(captions, orient="index").reset_index()
    df.columns = ['file_name', 'text']
    df.to_csv(f"{FACES_FOLDER}/metadata.csv", index=False)

    return {"message": "✅ Captions generated and saved"}


# TRAIN LORA MODEL
@app.post("/train_lora/")
async def train_lora():
    """Fine-tunes LoRA model using the captioned dataset."""
    train_script = "train_text_to_image_lora_sdxl.py"
    
    if not os.path.exists(train_script):
        raise HTTPException(status_code=404, detail="❌ Training script not found")
    
    command = [
        "accelerate", "launch", train_script,
        "--pretrained_model_name_or_path", "stabilityai/stable-diffusion-xl-base-1.0",
        "--pretrained_vae_model_name_or_path", "madebyollin/sdxl-vae-fp16-fix",
        "--train_data_dir", "faces",
        "--caption_column", "text",
        "--resolution", "1024",
        "--random_flip",
        "--train_batch_size", "1",
        "--num_train_epochs", "10",
        "--checkpointing_steps", "100",
        "--learning_rate", "1e-04",
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--mixed_precision", "fp16",
        "--seed", "404",
        "--output_dir", "sdxl-faces",
        "--validation_epochs", "2",
        "--validation_prompt", "A picture of a product",
        "--resume_from_checkpoint", "latest"
    ]
    
    subprocess.run(command)
    
    return {"message": "🚀 LoRA fine-tuning complete!"}


# IMAGE GENERATION WITH FINE-TUNED MODEL
class ImageGenRequest(BaseModel):
    prompt: str
    num_steps: int = 30
    seed: int = 42

@app.post("/generate/")
async def generate_image(request: ImageGenRequest):
    """Generates an image using the fine-tuned LoRA model."""
    if not pipeline:
        raise HTTPException(status_code=500, detail="❌ LoRA pipeline is not loaded")

    generator = torch.Generator(device=device).manual_seed(request.seed)
    generated_image = pipeline(
        prompt=request.prompt, num_inference_steps=request.num_steps, generator=generator
    ).images[0]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"generated_{timestamp}.png"
    generated_image.save(output_path)

    return {"message": "✅ Image generated", "filename": output_path}


# LIST UPLOADED & GENERATED IMAGES
@app.get("/list_images/")
async def list_uploaded_images():
    """Lists all uploaded images."""
    images = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith((".png", ".jpg", ".jpeg"))]
    return {"uploaded_images": images}

@app.get("/list_generated/")
async def list_generated_images():
    """Lists all generated images."""
    images = [f for f in os.listdir(".") if f.startswith("generated_") and f.endswith(".png")]
    return {"generated_images": images}


# DELETE IMAGE
@app.delete("/delete/{filename}")
async def delete_image(filename: str):
    """Deletes an image."""
    file_path = os.path.join(IMAGE_FOLDER, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"message": f"✅ {filename} deleted"}
    else:
        raise HTTPException(status_code=404, detail="❌ File not found")


# Store scheduled posts temporarily (In-Memory)
scheduled_posts = []

class SchedulePostRequest(BaseModel):
    date: str
    topic: str

@app.post("/schedule_post/")
async def schedule_post(request: SchedulePostRequest, background_tasks: BackgroundTasks):
    """Schedules a post and runs the full AI pipeline (scraping, captioning, LoRA training, generation)."""
    
    post = {"date": request.date, "topic": request.topic, "status": "Processing..."}
    scheduled_posts.append(post)

    # Run full AI pipeline asynchronously
    background_tasks.add_task(run_full_pipeline, request.topic)

    return {"message": "✅ Post scheduled successfully!", "scheduled_post": post}


import torch

def run_full_pipeline(topic: str):
    """Runs scraping, captioning, LoRA fine-tuning, and image generation for a scheduled post."""
    
    print(f"🚀 Starting AI pipeline for topic: {topic}")

    # 1️⃣ **Scrape Images**
    google_crawler = GoogleImageCrawler(storage={'root_dir': IMAGE_FOLDER})
    google_crawler.crawl(keyword=topic, max_num=10)
    print(f"✅ Scraped images for '{topic}'")

    # 2️⃣ **Resize Images**
    for file_name in os.listdir(IMAGE_FOLDER):
        file_path = os.path.join(IMAGE_FOLDER, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(file_path).convert("RGB")
            img = img.resize((1024, 1024))
            img.save(os.path.join(RESIZED_FOLDER, file_name))
    print("✅ Resized images")

    # 3️⃣ **Generate Captions with BLIP-2** (Inference Mode ✅)
    captions = {}
    with torch.inference_mode():  # ✅ Optimized for GPU inference
        for file_name in os.listdir(RESIZED_FOLDER):
            file_path = os.path.join(RESIZED_FOLDER, file_name)
            img = Image.open(file_path)
            caption = captioner(img)[0]['generated_text'].strip()
            captions[file_name] = caption
            img.save(os.path.join(FACES_FOLDER, file_name))

    # Save captions to CSV
    df = pd.DataFrame.from_dict(captions, orient="index").reset_index()
    df.columns = ['file_name', 'text']
    df.to_csv(f"{FACES_FOLDER}/metadata.csv", index=False)
    print("✅ Generated captions")

    # 4️⃣ **Train LoRA Model** (🚨 No `torch.inference_mode()` here!)
    train_script = "train_text_to_image_lora_sdxl.py"
    if not os.path.exists(train_script):
        print("❌ Training script not found")
        return

    command = [
        "accelerate", "launch", train_script,
        "--pretrained_model_name_or_path", "stabilityai/stable-diffusion-xl-base-1.0",
        "--pretrained_vae_model_name_or_path", "madebyollin/sdxl-vae-fp16-fix",
        "--train_data_dir", "faces",
        "--caption_column", "text",
        "--resolution", "1024",
        "--random_flip",
        "--train_batch_size", "1",
        "--num_train_epochs", "10",
        "--checkpointing_steps", "100",
        "--learning_rate", "1e-04",
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--mixed_precision", "fp16",
        "--seed", "404",
        "--output_dir", "sdxl-faces",
        "--validation_epochs", "2",
        "--validation_prompt", topic,
        "--resume_from_checkpoint", "latest"
    ]
    subprocess.run(command)
    print("✅ Fine-tuned LoRA model")

    # 5️⃣ **Generate an AI Image** (Inference Mode ✅)
    with torch.inference_mode():  # ✅ Optimized for GPU inference
        generator = torch.Generator(device=device).manual_seed(42)
        generated_image = pipeline(
            prompt=topic, num_inference_steps=30, generator=generator
        ).images[0]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"generated_{timestamp}.png"
    generated_image.save(output_path)
    print(f"✅ Generated AI image: {output_path}")

    # ✅ **Update the scheduled post with the generated image**
    for post in scheduled_posts:
        if post["topic"] == topic:
            post["status"] = "✅ Completed"
            post["generated_image"] = output_path




@app.get("/list_scheduled_posts/")
async def list_scheduled_posts():
    """Returns all scheduled posts."""
    return {"scheduled_posts": scheduled_posts}


# RUN FASTAPI SERVER
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
