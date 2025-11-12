import yaml
import os
import json
from sim.window import Window

with open("config.yaml") as f:
    params = yaml.safe_load(f)

# ==== Selección de modelo ==========
trained_dir = "models/trained"
trained_models = (
    sorted([f for f in os.listdir(trained_dir) if f.endswith(".npy")])
    if os.path.exists(trained_dir)
    else []
)

print("Seleccione modelo:")
print("[0] follow mouse")
for i, m in enumerate(trained_models, start=1):
    print(f"[{i}] {m}")

choice = int(input("> "))

if choice == 0:
    params["model"] = "followMouse"
else:  # seleccionar modelo de la carpeta models/trained
    params["model"] = "NN"
    base_name = trained_models[choice - 1]
    modelo_path = os.path.join(trained_dir, base_name)
    json_path = modelo_path.replace(".npy", ".json")
    with open(json_path) as f:
        saved_params = json.load(f)
    params.update(saved_params)
    params["modelo_path"] = modelo_path
# ===================================

Window(params).run()
