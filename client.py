import requests
import argparse
import os
import json

def createRequest(config_filepath, device_type="orin", timeout_increase=2):
    '''
    Function to read a configuration file containing the API endpoint, parameters to pass and receiving the response data.

    In case of a timeout error, the time taken for execution will be increased by a factor specified in timeout_increase.

    A maximum of 20 minutes (1200 seconds) is the maximum time provided for the API connection. If the timeout increases beyond 1200 seconds, the client will stop.

    Input Parameters:
        1. config_filepath  => Path to the configuration file.
        2. timeout_increase => Factor by which to increase the timeout time.

    Output:
        1. Response JSON with status codes 401, 504 and 200.
        2. For any other error, error message, JSON response from API, and status code from API response.
    '''

    config_filepath = os.path.abspath(config_filepath)

    with open(config_filepath, "r") as in_file:
        contents = json.load(in_file)
        pass

    api_endpoint = contents["url"]
    params = contents["params"]
    model_filepath = os.path.abspath(contents["model_filepath"])
    model_files = {"model_file" : (os.path.basename(model_filepath), open(model_filepath, "rb"))} 

    if device_type == "orin":
        model_files = [("model_file", (os.path.basename(model_filepath), open(model_filepath, "rb")))]
    elif device_type == "myriad":
        model_bin_filepath = os.path.splitext(model_filepath)[0] + ".bin"
        model_files = [("model_file", (os.path.basename(model_filepath), open(model_filepath, "rb"))), ("model_file", (os.path.basename(model_bin_filepath), open(model_bin_filepath, "rb")))]
    else:
        print(f"[ERROR] Device type not supported. Current selection : {device_type}")
        pass
    
    while params["timeout"] <1200:

        response = requests.post(api_endpoint, params=params, files=model_files)

        if response.status_code == 401:
            print("Wrong API key")
            return response.json()
            break
        elif response.status_code == 504:
            
            # print(response.json())
            params["timeout"] *= timeout_increase
        elif response.status_code == 200:

            # print(response.json())
            return response.json()
        else:
            print("Check error message. Error response is passed as default return for this function")
            break
        pass

    return {"error" : "Process took more than 20 minutes to run or failed.", "result":response.json(), "code" : response.status_code}
    pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--config_file", required=True,
                    help="Path to the config file")
    args = vars(ap.parse_args())

    print(createRequest(config_filepath=args["config_file"]))
    pass