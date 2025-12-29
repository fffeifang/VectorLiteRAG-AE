import os
import json
from pathlib import Path

configs_dir = Path(__file__).resolve().parent

def write_json(path, data):
    path = os.path.join(configs_dir, path)
    tmp = path + '.tmp'
    with open(tmp, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.replace(tmp, path)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_model(model_name):
    path = os.path.join(configs_dir, "models", f"{model_name}.json")
    return load_json(path)

def load_all_models():
    models_dir = Path(configs_dir) / 'models'
    json_files = [f for f in models_dir.iterdir() if f.suffix == '.json']
    models = {str(f.name).replace('.json',''): load_json(str(f)) for f in json_files}
    return models

def load_index():
    path = os.path.join(configs_dir, "index.json")
    return load_json(path)

def load_gpu():
    path = os.path.join(configs_dir, "gpu.json")
    return load_json(path)

def load_all_configs(model_name):
    model = load_model(model_name)
    index = load_index()
    gpu = laod_gpu()

    return {
        "model": model,
        "index": index,
        "gpu": gpu
    }