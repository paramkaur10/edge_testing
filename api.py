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
from typing import Optional
import xml.etree.ElementTree as ET

# import torch
# import torch.nn.functional as F
import numpy as np
import asyncio

from openvino.runtime import Core

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

        for i in range(num_classes):
            pred_mask = (preds == i)
            target_mask = (targets == i)

            intersection = (pred_mask & target_mask).float().sum(dim=(1, 2))
            union = (pred_mask | target_mask).float().sum(dim=(1, 2))

            iou = (intersection + 1e-6) / (union + 1e-6)
            ious.append(iou)

        ious = torch.stack(ious, dim=0)  # [num_classes, B]
        mean_iou = ious.mean(dim=0).mean().item()
        return mean_iou
        pass
    pass


def calculateIOUNumpy(logits, targets, num_classes):

    preds = np.argmax(logits, axis=1)
    ious = []

    for i in range(num_classes):
        pred_mask = (pred == i)
        target_mask = (targets == i)

        intersection = (pred_mask & target_mask).sum(axis=(1,2))
        union = (pred_mask | target_mask).sum(axis=(1,2))

        iou = (intersection + 1e-6) / (union + 1e-6)
        ious.append(iou)
        pass

    ious = np.stack(ious, axis=0)
    mean_iou = np.mean(ious, axis=0)
    return mean_iou
    pass

def load_modelOrin(model_path: str, device_preference: str="GPU"):
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

def loadModelMyriad(model_path: str, batch_size=1, input_shape=(7,256,256), device_preference: str="MYRIAD"):
    '''
    Function to load the model on the Myriad X
    '''

    assert os.path.exists(model_path), f"Model does not exist at {model_path}"

    ie = Core()
    model_ir = ie.read_model(model=model_path)
    input_layer_ir = model_ir.input(0)
    print(input_layer_ir.shape)

    new_shape = (batch_size, input_shape[0], input_shape[1], input_shape[2])

    model_ir.reshape({input_layer_ir.any_name : new_shape})

    input_layer = model_ir.input(0)
    print(input_layer.shape)

    compiled_model_ir = ie.compile_model(model=model_ir, device_name=device_preference)
    output_layer_ir = compiled_model_ir.output(0)

    return compiled_model_ir, output_layer_ir
    pass

async def runInferenceOrin(model_path, device_type="orin"):
    print("[INFO Model Loading] Loading the model")

    model, model_device = load_modelOrin(model_path, device_preference="GPU")

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
        pass

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
    pass

async def runInferenceMyriad(model_path, device_type="myriad", input_shape=(-1, 7,256,256)):
    '''
    Asynchronus function for running inference on Myriad X

    Input Parameters:
        1. model_path  => Path to the XML model
        2. input_shape => Tuple of 4 elements; B, H, W, C . TO BE CHANGED BY THE PYNAS TEAM.

    Outputs:
    '''

    print("[INFO Model Loading] Loading the model on the Myriad X")

    compiled_model, output_layer = loadModelMyriad(model_path=model_path)

    test_dir = '/home/PyNasResearch/test_dataset' 
    test_images_path = os.path.join(test_dir, 'numpy_images')
    preprocessed_image_files = glob(f'{test_images_path}/*.npy')
    print('[INFO] Grabbed preprocessed images')

    image_tiles = []
    mask_tiles = []
    for pi in preprocessed_image_files[:1]:
        pm = pi.replace('image', 'mask')
        assert os.path.exists(pi) and os.path.exists(pm)
        image_tiles.append(np.load(pi))
        mask_tiles.append(np.load(pm))
        pass
    print('[INFO] Loaded images and masks')

    image_tensor = np.stack(image_tiles)
    input_data = np.reshape(image_tensor, input_shape)

    output_predictions = []
    targets = []
    print("[INFO] Starting Inference.")
    inf_start_time = time.time()
    for i in range(0, input_data.shape[0]):

        batch_image = np.expand_dims(input_data[i], axis=-1)
        print(batch_image.shape)
        predictions = compiled_model([batch_image])[output_layer]

        output_predictions.extend(predictions)

        targets.append(np.argmax(mask_tiles[i], axis=-1))
        pass

    inf_end_time = time.time()

    inference_time = inf_end_time - inf_start_time
    fps = input_data.shape[0] / inference_time

    print(f"[INFO] Inference Complete. FPS {fps:.2f}")

    # Add IOU function here for MyriadX board.
    output_predictions = np.stack(output_predictions)
    targets = np.stack(targets)
    mean_iou = calculateIOUNumpy(logits=output_predictions, targets=targets, num_classes=input_data.shape[1])
    # result = {"accuracy" : str(mean_iou), "latency" : str(fps)}
    result = {"accuracy" : str(0), "latency" : str(10)}

    return 0, result
    pass


# , model_bin: Optional[UploadFile] = None
@app.post("/secure")
async def checkSecure(api_key: str, timeout: int, device: str, model_file: UploadFile = File(...), 
                      model_bin: Optional[UploadFile] = None):
    '''
    Function to check if the API Token authtentication works.
    '''
    user = get_user(api_key)
    timeout = int(timeout)


    import random
    async def long_running_task():
        await asyncio.sleep(5)  # sleep for 10 seconds
        return random.random()  # return a random number
        pass

    print(model_file)
    # print(model_file.filename)
    # print(model_bin)
    model_path = UPLOAD_DIR / model_file.filename

    if device=="myriad":
        if not os.path.exists(model_path):
            xml_content = await model_file.read()
            root = ET.fromstring(xml_content)
            # image_element = ET.SubElement(root, 'image')
            tree = ET.ElementTree(root)

            with open(model_path, "wb") as mf:
                tree.write(mf, encoding='utf-8', xml_declaration=True)
                pass
            pass
        pass

    if model_bin:
        print(model_bin)
        model_bin_path = UPLOAD_DIR/ model_bin.filename
        with open(model_bin_path, "wb") as mf:
            mf.write(await model_bin.read())
            pass
        pass

    cache = {"result" : random.random()}
    try:
        result_returncode, result = await asyncio.wait_for(runInferenceMyriad(model_path=model_path), timeout=timeout)  # timeout after five seconds of waiting
        cache["result"] = result  # cache the result
        response = {"message": result, "cache": False}
        status = 200
        return response
    except TimeoutError as e:
        response = {"error" : "Timeout", "Current timeout value" : timeout}
        status = 504
    except Exception as e:
        response = {"error" : "An error has occured", "Current timeout value" : timeout, "Error" : e}
        status = 500
        pass

    # result_returncode, result = await asyncio.wait_for(runInferenceMyriad(model_path=model_path, device_type=device), timeout=timeout)  
    # response = {"message" : result}
    # status = 200
    # print(response)
    return JSONResponse(content=response, status_code=status)
    pass

@app.post("/run-inference/")
async def run_inference(
    timeout: int,
    api_key: str,
    device: str,
    model_file: UploadFile = File(...),
    model_bin: Optional[UploadFile] = None
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
        print(model_file.filename)
        model_path = UPLOAD_DIR / model_file.filename

        if device=="myriad" and not os.path.exists(model_path):
            xml_content = await model_file.read()
            root = ET.fromstring(xml_content)
            # image_element = ET.SubElement(root, 'image')
            tree = ET.ElementTree(root)

            with open(model_path, "wb") as mf:
                tree.write(mf, encoding='utf-8', xml_declaration=True)
                pass
            pass
        else:
            with open(model_path, "wb") as mf:
                mf.write(await model_file.read())
                pass
            pass

        if model_bin:
            model_bin_path = UPLOAD_DIR / model_bin.filename
            with open(model_bin_path, "wb") as mf:
                mf.write(await model_bin.read())
                pass
            pass
        
        try:
            if device == "orin":
                result_returncode, result = await asyncio.wait_for(runInferenceOrin(model_path=model_path, device_type=device), timeout=timeout)
            elif device == "myriad":
                result_returncode, result = await asyncio.wait_for(runInferenceMyriad(model_path=model_path, device_type=device), timeout=timeout)
            else:
                result_returncode = 1
                result = "Incorrect device type provided"
                pass
            print(result_returncode)
        except TimeoutError as e:
            return JSONResponce(content={"error" : "A timeout has occured", "details" : f"Timeout at {timeout} seconds"}, status=504)
        except Exception as e:
            return JSONResponse(content={"error": "An Error has occured", "details" : str(e)}, status_code=500)
            pass

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