import matplotlib.pyplot as plt
import numpy as np
import os

# ===== eleccion de historial =====
trained_dir = "models/trained"
trained_models = (
    sorted([f for f in os.listdir(trained_dir) if f.endswith("-history.npy")])
    if os.path.exists(trained_dir)
    else []
)

for i, m in enumerate(trained_models):
    print(f"[{i}] {m}")

choice = int(input("> "))

# ===== cargar historial =====

base_name = trained_models[choice - 1]
history_path = os.path.join(trained_dir, base_name)
fit_history = np.load(history_path)
avg = fit_history[:, 0]
mejor = fit_history[:, 1]

plt.grid()
plt.plot(avg, label="avg")
plt.plot(mejor, label="mejor")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.show()
