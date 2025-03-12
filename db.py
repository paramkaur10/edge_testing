import json

# api_keys = {
#     "e54d4431-5dab-474e-b71a-0db1fcb9e659": "7oDYjo3d9r58EJKYi5x4E8",
#     "5f0c7127-3be9-4488-b801-c7b6415b45e9": "mUP7PpTHmFAkxcQLWKMY8t"
# }

# users = {
#     "7oDYjo3d9r58EJKYi5x4E8": {
#         "name": "Bob"
#     },
#     "mUP7PpTHmFAkxcQLWKMY8t": {
#         "name": "Alice"
#     },
# }

# Add code for reading from a JSON for API Keys and Uers

with open("ServerFiles/apikeys.json", "r") as api_key_json:
    api_keys = json.load(api_key_json)
    pass

with open("ServerFiles/serverusers.json") as users_json:
    users = json.load(users_json)
    pass

def check_api_key(api_key: str):
    return api_key in api_keys
    pass

def get_user_from_api_key(api_key: str):
    return users[api_keys[api_key]]
    pass