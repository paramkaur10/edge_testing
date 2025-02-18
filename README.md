# FastAPI Inference API

This project provides a FastAPI-based API to upload a single model file, run an inference script (`inference.py`), and return the results.

## Features

- Accepts a single file upload via a `POST` request:
  - **Model file** (e.g., a complete file containing both weights and architecture).
- Executes the `inference.py` script with the uploaded file.
- Returns the inference results or an error message.

## Requirements

- Python 3.10
- FastAPI
- Uvicorn
- Other dependencies as required by your `inference.py`.

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:LittlePlaceLabs/lpl-PyNas-EdgeTesting.git
   cd API
   ```

2. Set up a virtual environment and activate it (or use miniconda):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the API

1. Start the FastAPI server:
   ```bash
   uvicorn api:app --reload
   ```
   The API will be available at `http://127.0.0.1:8000`.

### API Endpoint

#### `POST /run-inference/`

Uploads a single model file and runs inference.

- **Request**:
  - `model_file`: File to be uploaded (e.g., a complete model file).

  Example `cURL` command (from Client):
  ```bash
  curl -X POST "http://127.0.0.1:8000/run-inference/" \
       -F "model_file=@path_to_model_file"
  ```

- **Response** (example):
  - On success:
    ```json
    {
      "result": "Inference output"
    }
    ```
  - On failure:
    ```json
    {
      "error": "Error message",
      "details": "Additional error details"
    }
    ```

## Example `inference.py`

The `inference.py` script should process the single model file. 

To be **tailored** to a specific edge device.

## Folder Structure

```
.
├── app.py                # FastAPI script
├── inference.py          # Inference logic script
├── uploaded_files/       # Directory for uploaded files
├── requirements.txt      # Python dependencies
└── README.md             # Documentation
```

## Dependencies

List of required packages (add to `requirements.txt`):
- `fastapi`
- `uvicorn`

Install them with:
```bash
pip install fastapi uvicorn
```

## Notes

- Ensure `inference.py` is in the project directory and executable.
- The uploaded files will be stored in the `uploaded_files/` directory. Clean this directory periodically if needed.
