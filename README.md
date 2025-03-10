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

3. Install dependencies (Server Side):
   ```bash
   pip install -r server-requirements.txt
   ```

4. Install dependencies (Client Side):
    ```bash
    pip install -r client-requirements.txt
    ```

## Usage

### Creating an API Token
This step is necessary before starting a fresh server. After a fresh server, this command is only used for adding new users.

`python createauthtoken.py --username <user_name>`

The above step will create a folder called _ServerFiles_, which will contain two JSON files, *apikeys.json* and *serverusers.json*, and text files corresponding to the user name(s) provided which will contain the API auth token. 

The text files are to be shared with the client. The JSON files are not to be shared at all.

### For the Client
#### Creating the Configuration File
A client/user will have to create a configuration file under the folder _configs_ for the `client.py` python program. The fields are explained below

1. `url` : The API endpoint including the port number
2. `params` : Contains the parameters for the API
  1. `api_key` : Client's provided API token.
  2. `timeout` : Number of seconds after which the API connection will timeout.
3. `model_filepath` : Path to the model to be used by the API.

#### Running the client program
There are two ways for the client to run the program, from the command line and from a function call.

1. Command line : `python client.py --config_file <path_to_config_file>`

2. Function Call : `client.createRequest(config_filepath, timeout_increase=2)`

The program will return the JSON of the response from the API if there are no errors. If the API key is wrong, a 403 along with the corresponding JSON will be returned.

In case of 504, timeout, the program will automatically increase the timeout time by a factor specified in `timeout_increase` parameter. There is a hardstop at 1200 seconds (20 minutes).

In case of no error, a 200 status code and the JSON response will be returned.

In case of any other error, the corresponding error code and JSON response will be returned. 

### Running the API

1. Start the FastAPI server:
   ```bash
   uvicorn api:app --reload --reload-include *.json --port 9000 --host 0.0.0.0
   ```

   For MAC Users, use the following command : 
   ```bash
   uvicorn api:app --reload --reload-include "*.json" --port 9000 --host 0.0.0.0
   ```

   The API will be available at `http://0.0.0.0:9000`.

### API Endpoint

#### `POST /run-inference/`

Uploads a single model file and runs inference.

- **Request**:
  - `model_file`: File to be uploaded (e.g., a complete model file).
  - `api_key` : Parameter containing the API Token for authentication.
  - `timeout` : Parameter to decide when to terminate the API connection in seconds

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
