import numpy as np
from tqdm import tqdm
from training.evaluador import Evaluador

# ============================================================
# ======================== PARAMETROS ========================
# ============================================================
rng: np.random.Generator = np.random.default_rng()
params = {
    # ESCENARIO
    "w_w": 300,  # ancho
    "w_h": 300,  # alto
    "obj_r": 25,  # radio del objetivo
    # OVEJAS
    "N": 50,  # numero de ovejas
    "r_s": 65.0,  # radio de repulsion del pastor
    "r_a": 2.0,  # radio de repulsion de otras ovejas
    "h": 0.5,  # coeficiente de inercia
    "c": 1.05,  # coeficiente de cohesion
    "rho_a": 2.0,  # fuerza de repulsion (oveja-oveja)
    "rho_s": 1.0,  # fuerza de repulsion (pastor-oveja)
    "e": 0.3,  # ruido angular (componente estocastica)
    "delta": 1.0,  # distancia por paso
    "n_neigh": 7,  # numero de vecinos para cohesion
    "r_walk": 0.05,  # probabilidad de random walk
    # PASTOR
    "p_delta": 1.5,  # distancia por paso
    "model": "NN",  # modelo de comportamiento del pastor (followMouse | NN)
    "min_w": -64,
    "max_w": 64,
    "pers_ovejas": 5,  # ovejas percibidas por el pastor
    "pers_pastores": 0,  # otros pastores percibidos por el pastor
    "hidden_lay_1": 8,  # neuronas de la primer capa oculta
    "hidden_lay_2": 4,  # neuronas de la segunda capa oculta
    # ENTRENAMIENTO
    "max_steps": 2000,  # numero maximo de pasos por simulacion
    "poblacion": 5,  # individuos por poblacion
    "generaciones": 5,  # generaciones
    "progenitores": 2,  # progenitores por generacion
    "mutacion": 0.05,  # probabilidad de mutacion
}
# ==========================================================

evaluador = Evaluador(params, rng)
fit_history = np.empty((0, 2), dtype=float)
ventana = params["poblacion"] // params["progenitores"]
n_inputs = (
    params["pers_ovejas"] + params["pers_pastores"] + 2
)  # ovejas, otros pastores, el objetivo y su propia posicion
n_bits = 8 * (
    (
        n_inputs * params["hidden_lay_1"] + params["hidden_lay_1"]
    )  # capa oculta 1: por cada neurona, un peso por input y un bias
    + (
        params["hidden_lay_1"] * params["hidden_lay_2"] + params["hidden_lay_2"]
    )  # capa oculta 2
    + (params["hidden_lay_2"] * 2 + 2)  # capa de salida: 2 neuronas de salida, FIJO
)

# 1) inicializar la poblacion al azar
poblacion = rng.integers(0, 2, (params["poblacion"], n_bits), dtype=np.uint8)

# 2) calcular fitness
fit = [
    evaluador.evaluar(poblacion[i]) for i in range(params["poblacion"])
]  # TODO: paralelizar
sorted = np.argsort(fit)  # indices que ordenan de menor a mayor

c_tol: int = 0  # contador de generaciones sin mejora
fit_elite = fit[sorted[-1]]
fit_history = np.vstack([fit_history, [0, fit_elite]])
for g in tqdm(range(params["generaciones"])):
    # 1) elegir progenitores: un elite y el resto por ventana
    progenitores = np.empty((params["progenitores"], n_bits), dtype=np.uint8)
    progenitores[0] = poblacion[sorted[-1]]  # elite

    v = ventana
    for i in range(1, params["progenitores"]):
        progenitores[i] = poblacion[sorted[-rng.integers(0, v, 1)]]
        v += ventana

    # 2) cruzar progenitores (cruza simple)
    poblacion[: params["progenitores"]] = progenitores
    for i in range(params["progenitores"], params["poblacion"]):
        p1, p2 = rng.integers(
            0, params["progenitores"], 2
        )  # tomar progenitores al azar

        # punto de cruza: elegir inicio y fin del segmento a cruzar
        c1, c2 = np.sort(rng.integers(0, n_bits, 2))

        # cruzar: segmento del padre 1 y el resto del padre 2
        poblacion[i, :c1] = progenitores[p2, :c1]  # antes de c1
        poblacion[i, c1:c2] = progenitores[p1, c1:c2]  # c1 a c2
        poblacion[i, c2:] = progenitores[p2, c2:]  # de c2 al final

        # mutar
        if rng.random() < params["mutacion"]:
            b = rng.integers(0, n_bits)
            poblacion[i, b] ^= 1  # invierte 0 a 1 y viceversa

    # 3) evaluar fitness
    fit = [
        evaluador.evaluar(poblacion[i]) for i in range(params["poblacion"])
    ]  # TODO: paralelizar
    sorted = np.argsort(fit)  # indices que ordenan de menor a mayor
    if fit[sorted[-1]] > fit_elite:
        c_tol = 0
        fit_elite = fit[sorted[-1]]
        fit_history = np.vstack([fit_history, [g + 1, fit_elite]])

print(fit_elite)
print(fit_history)
