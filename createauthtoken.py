'''
File for creating an API auth token for a user and storing it on the server.
'''

# api_keys = {
#     "e54d4431-5dab-474e-b71a-0db1fcb9e659": "7oDYjo3d9r58EJKYi5x4E8"
# }

# users = {
#     "7oDYjo3d9r58EJKYi5x4E8": {
#         "name": "Bob"
#     }
# }

import json
import os
import argparse
import secrets

from base64 import b64encode, b64decode
from pathlib import Path

def createToken(username):
    '''
    Function for creating the API auth token and linking to the user.
    '''

    api_key = secrets.token_urlsafe(16)

    # Create the folder for containing any relevant server files. This should not be accessible from the API calls.
    if not os.path.exists("ServerFiles"):
        os.makedirs("ServerFiles")

        with open("ServerFiles/apikeys.json", "w") as out_file:
            json.dump({}, out_file)
            pass

        with open("ServerFiles/serverusers.json", "w") as out_file:
            json.dump({}, out_file)
            pass
        pass

    uuid = os.urandom(8)
    uuid = b64encode(uuid).decode('utf-8')

    with open("ServerFiles/apikeys.json", "r") as in_file:
        api_keys_dict = json.load(in_file)
        pass

    with open("ServerFiles/serverusers.json", "r") as in_file:
        users_dict = json.load(in_file)
        pass

    api_keys_dict[api_key] = uuid
    users_dict[uuid] = username

    with open("ServerFiles/apikeys.json", "w") as out_file:
        json.dump(api_keys_dict, out_file, indent=4)
        pass

    with open("ServerFiles/serverusers.json", "w") as out_file:
        json.dump(users_dict, out_file, indent=4)
        pass

    with open(f"ServerFiles/{username}.txt", "w") as out_file:
        out_file.write(api_key)
        print(f"[LPL INFO] Key stored in ServerFiles/{username}.txt")
        pass
    pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-u","--username", required=True,
                    help="Username for connecting client.")
    args = vars(ap.parse_args())

    createToken(username=args["username"])
    pass