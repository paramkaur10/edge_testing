from fastapi import FastAPI, File, UploadFile, Depends
from fastapi import Security, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
import subprocess
import os
import json
from pathlib import Path

import torch

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
OUTPUT_DIR = ORBITFY_DIR / "outbox/outbox-rt-cloudzero"  # Where results are saved

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Serve static files (for masks, logs, reports, etc.)
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

async def runCloudZero(model_path, device_preference="GPU"):
    '''
    Running CloudZero model through this function.
    ''' 

    print("[INFO Model Loading] Loading the model")

    model = torch.load(model_path)

    model.eval()

    gpu_available = torch.cuda.is_available()

    if device_preference == "GPU" and gpu_available:
        device = torch.device('cuda')
        try:
            model.to(device)
            model_device = "GPU"
            print("[INFO Model Loading] Model loaded to the GPU.")
        except Exception as e:
            summary_logger.warning(f"CUDA is available, but model could not be moved to GPU: {e}")
            device = torch.device('cpu')
            model_device = "CPU"
    elif device_preference == "GPU" and not gpu_available:
        summary_logger.warning("GPU requested but CUDA is not available. Falling back to CPU.")
        device = torch.device('cpu')
        model_device = "CPU"
    elif device_preference == "CPU":
        device = torch.device('cpu')
        model_device = "CPU"
    else:
        summary_logger.warning(f"Invalid device preference '{device_preference}'. Defaulting to CPU.")
        device = torch.device('cpu')
        model_device = "CPU"
        pass

    # result = await proc.communicate()

    # print(proc)
    # print(result)
    return 1, "Successful model loading"
    pass

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
    Uploads a model file, runs orbitfy_cloudzero_executor.py, and returns all outputs.

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
            result_returncode, result = await asyncio.wait_for(runCloudZero(model_path=model_path), timeout=timeout)
            print(result_returncode)
        except Exception as e:
            return JSONResponse(content={"error": "Timeout error", "details" : f"Timed out at {timeout} seconds"},
                                status_code=504)
            pass

        # Check if execution failed
        if result_returncode != 0:
            return JSONResponse(
                content={"error": "Execution failed", "details": result.stderr},
                status_code=500,
            )

        # Collect Output Files
        confidence_score = None
        mask_files = []
        json_reports = []
        log_files = []

        for file in os.listdir(OUTPUT_DIR):
            file_path = f"/static/{file}"

            if file.endswith(".png"):  # Segmentation masks
                mask_files.append(file_path)

            elif file.endswith(".json"):  # JSON reports
                json_reports.append(file_path)

                # Try to extract confidence from JSON reports
                with open(os.path.join(OUTPUT_DIR, file), "r") as json_file:
                    try:
                        data = json.load(json_file)
                        if "confidence_percentage" in data:
                            confidence_score = data["confidence_percentage"]
                    except json.JSONDecodeError:
                        pass

            elif file.endswith(".log"):  # Log files
                log_files.append(file_path)

        # Prepare API response
        response_data = {
            "confidence_percentage": confidence_score,
            "segmentation_masks": mask_files,
            "json_reports": json_reports,
            "logs": log_files,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

        return JSONResponse(content=response_data, status_code=200)

    except Exception as e:
        return JSONResponse(
            content={"error": "An error occurred", "details": str(e)},
            status_code=500,
        )