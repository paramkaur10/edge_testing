from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import subprocess
import os
import json
from pathlib import Path

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


@app.post("/run-inference/")
async def run_inference(
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
    try:
        # Save uploaded model file to 'uploaded_files'
        model_path = UPLOAD_DIR / model_file.filename
        with open(model_path, "wb") as mf:
            mf.write(await model_file.read())

        # Run Orbitfy Executor (orbitfy_cloudzero_executor.py)
        result = subprocess.run(
            ["python3", str(SCRIPT_DIR / "orbitfy_cloudzero_executor.py"), CONFIG_FILE],
            cwd=SCRIPT_DIR,  # Set working directory to the scripts folder
            capture_output=True,
            text=True,
        )

        # Check if execution failed
        if result.returncode != 0:
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
