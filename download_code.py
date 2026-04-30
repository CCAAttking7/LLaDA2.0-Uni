import os
import urllib.request
import json

repo_id = "inclusionAI/LLaDA2.0-Uni"
api_url = f"https://hf-mirror.com/api/models/{repo_id}"
os.makedirs("llada_code", exist_ok=True)

try:
    req = urllib.request.Request(api_url)
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        siblings = data.get("siblings", [])
        for file_info in siblings:
            filename = file_info["rfilename"]
            if filename.endswith(".py") or filename.endswith(".json"):
                # We only need Python code and config files
                if filename.endswith(".json") and "config" not in filename: 
                    continue
                print(f"Downloading {filename}...")
                dl_url = f"https://hf-mirror.com/{repo_id}/resolve/main/{filename}"
                # Handle nested directories if any
                os.makedirs(os.path.join("llada_code", os.path.dirname(filename)), exist_ok=True)
                urllib.request.urlretrieve(dl_url, os.path.join("llada_code", filename))
    print("Done! Code downloaded to llada_code/")
except Exception as e:
    print(f"Failed: {e}")
