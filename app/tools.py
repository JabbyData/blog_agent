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
    dir_path = os.getcwd()
    configs_path = os.path.join(dir_path, "configs.json")
    print(os.path.join(dir_path, "agents", "query_analyzer", "tokenizer.model.v3"))

    with open(configs_path, "r") as f:
        configs = json.load(f)

    try:
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
            configs["query_analyzer"] = {
                "tokenizer_path": os.path.join(
                    query_analyzer_dir, "tokenizer.model.v3"
                ),
                "model_path": os.path.join(query_analyzer_dir),
            }
            with open(configs_path, "w") as f:
                json.dump(configs, f)

    except Exception as e:
        if os.path.exists(query_analyzer_dir):
            shutil.rmtree(query_analyzer_dir)
        print(f"An error occured {e}")

    print("Models loaded !")


if __name__ == "__main__":
    _ = load_models()
