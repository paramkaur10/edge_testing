from fastapi import FastAPI, File, UploadFile, Depends
from fastapi import Security, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
import subprocess
import os
import json
from pathlib import Path
from glob import glob 
import time

import torch
import torch.nn.functional as F
import numpy as np
import asyncio

from auth import get_user

app = FastAPI()

# Directories
BASE_DIR = Path(__file__).resolve().parent
ORBITFY_DIR = BASE_DIR / "orbitfy-cloudzero"
SCRIPT_DIR = ORBITFY_DIR / "scripts"
CONFIG_FILE = ORBITFY_DIR / "config/cloudzero-config-template.json"

# I/O Directories
UPLOAD_DIR = BASE_DIR / "uploaded_files"
# OUTPUT_DIR = ORBITFY_DIR / "outbox/outbox-rt-cloudzero"  # Where results are saved

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# Serve static files (for masks, logs, reports, etc.)
# app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")



def calculate_iou(logits, targets, num_classes):
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)  # [B, H, W]
        ious = []

        for cls in range(num_classes):
            pred_mask = (preds == cls)
            target_mask = (targets == cls)

            intersection = (pred_mask & target_mask).float().sum(dim=(1, 2))
            union = (pred_mask | target_mask).float().sum(dim=(1, 2))

            iou = (intersection + 1e-6) / (union + 1e-6)
            ious.append(iou)

        ious = torch.stack(ious, dim=0)  # [num_classes, B]
        mean_iou = ious.mean(dim=0).mean().item()
        return mean_iou


def load_model(model_path: str,device_preference: str="GPU"):
    assert os.path.exists(model_path)
    model = torch.load(model_path)
    model.eval()

    gpu_available = torch.cuda.is_available()
    if device_preference == "GPU" and gpu_available:
        device = torch.device('cuda')
        try:
            model.to(device)
            model_device = "cuda"
            print("[INFO Model Loading] Model loaded to the GPU.")
        except Exception as e:
            print(f"[WARNING Model Loading] CUDA is available, but model could not be moved to GPU: {e}")
            device = torch.device('cpu')
            model_device = "cpu"
    elif device_preference == "GPU" and not gpu_available:
        print("[WARNING Model Loading] GPU requested but CUDA is not available. Falling back to CPU.")
        device = torch.device('cpu')
        model_device = "cpu"
    elif device_preference == "CPU":
        device = torch.device('cpu')
        model_device = "cpu"
    else:
        print(f"[WARNING Model Loading] Invalid device preference '{device_preference}'. Defaulting to CPU.")
        device = torch.device('cpu')
        model_device = "cpu"
    return model, model_device

async def runInference(model_path, device_preference="GPU"):
    print("[INFO Model Loading] Loading the model")
    model, model_device = load_model(model_path, device_preference)

    test_dir = '/home/PyNasResearch/test_dataset' 
    test_images_path = os.path.join(test_dir, 'numpy_images')
    preprocessed_image_files = glob(f'{test_images_path}/*.npy')
    print('[INFO] Grabbed preprocessed images')

    image_tiles = []
    mask_tiles = []
    for pi in preprocessed_image_files:
        pm = pi.replace('image', 'mask')
        assert os.path.exists(pi) and os.path.exists(pm)
        image_tiles.append(np.load(pi))
        mask_tiles.append(np.load(pm))
    print('[INFO] Loaded images and masks')

    logits_batch = []
    targets_batch = []

    print('[INFO] Starting inference')
    inf_start_time = time.time()
    for i, tile in enumerate(image_tiles):
        tile_tensor = torch.tensor(tile).float().permute(2, 0, 1).unsqueeze(0).to(model_device)

        with torch.no_grad():
            logits = model(tile_tensor)  # [1, C, H, W]
        logits_batch.append(logits.squeeze(0).cpu())  # [C, H, W]

        mask = mask_tiles[i]  # [H, W, C]
        mask_tensor = torch.argmax(torch.tensor(mask), dim=-1)  # [H, W]
        targets_batch.append(mask_tensor)

    inf_end_time = time.time()
    inference_time = inf_end_time - inf_start_time
    fps = len(image_tiles) / inference_time
    print(f'[INFO] Inference complete. FPS: {fps:.2f}')

    logits_tensor = torch.stack(logits_batch)   # [B, C, H, W]
    targets_tensor = torch.stack(targets_batch) # [B, H, W]
    num_classes = logits_tensor.shape[1]
    mean_iou = calculate_iou(logits_tensor, targets_tensor, num_classes)

    result = {'accuracy': str(mean_iou), 'latency': str(fps)}
    print(result)
    return 0, result


@app.post("/secure")
async def checkSecure(api_key: str, timeout: int, model_file: UploadFile = File(...)):
    '''
    Function to check if the API Token authtentication works.
    '''
    user = get_user(api_key)

    import random
    async def long_running_task():
        await asyncio.sleep(5)  # sleep for 10 seconds
        return random.random()  # return a random number
        pass

    cache = {"result" : random.random()}
    try:
        result = await asyncio.wait_for(long_running_task(), timeout=timeout)  # timeout after five seconds of waiting
        cache["result"] = result  # cache the result
        response = {"message": result, "cache": False}
        status = 200
        return response
    except Exception as e:
        response = {"error" : "A timeout error occured", "timeout" : timeout}
        status = 504
        pass

    return JSONResponse(content=response, status_code=status)
    pass

@app.post("/run-inference/")
async def run_inference(
    timeout: int,
    api_key: str,
    model_file: UploadFile = File(...),
):
    """
    Uploads a model file, runs inference, calculates accuracy and speed metrics, and returns mean iou and fps.

    Returns:
        JSON response containing:
        - Confidence %
        - Paths to segmentation mask
        - Logs and JSON reports
    """

    user = get_user(api_key)
    timeout = int(timeout)

    try:
        # Save uploaded model file to 'uploaded_files'
        model_path = UPLOAD_DIR / model_file.filename
        with open(model_path, "wb") as mf:
            mf.write(await model_file.read())

        try:
            result_returncode, result = await asyncio.wait_for(runInference(model_path=model_path), timeout=timeout)
            print(result_returncode)
        except Exception as e:
            return JSONResponse(content={"error": "Timeout error", "details" : f"Timed out at {timeout} seconds"},
                                status_code=504)

        # Check if execution failed
        if result_returncode != 0:
            return JSONResponse(
                content={"error": "Execution failed", "details": result},
                status_code=500,
            )

        return JSONResponse(content=result, status_code=200)

    except Exception as e:
        return JSONResponse(
            content={"error": "An error occurred", "details": str(e)},
            status_code=500,
        )