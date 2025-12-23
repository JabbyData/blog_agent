# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "huggingface-hub",
#     "pathlib",
# ]
# ///
import json
import os
import shutil

from huggingface_hub import snapshot_download


def load_models():
    paths = {}
    dir_path = os.getcwd()
    config_path = os.path.join(dir_path, "configs.json")
    with open(config_path, "r") as config_file:
        configs = json.load(config_file)

    try:
        hf_configs = configs["huggingface"]
        agents_dir = os.path.join(dir_path, "agents")

        ## Query Analyzer
        query_analyzer_dir = os.path.join(agents_dir, "query_analyzer")
        if not os.path.exists(query_analyzer_dir):
            print("Query analyzer does not exists, downloading model ...")
            os.makedirs(query_analyzer_dir)
            snapshot_download(
                repo_id="mistralai/Mistral-7B-Instruct-v0.3",
                allow_patterns=[
                    "params.json",
                    "consolidated.safetensors",
                    "tokenizer.model.v3",
                ],
                local_dir=query_analyzer_dir,
            )

    except Exception as e:
        if os.path.exists(query_analyzer_dir):
            shutil.rmtree(query_analyzer_dir)
        print(f"An error occured {e}")

    print("Models loaded !")
    return paths


if __name__ == "__main__":
    _ = load_models()
