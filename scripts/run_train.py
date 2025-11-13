import multiprocessing as mp
import logging
import yaml
import os
import time
import json
import numpy as np
from tqdm import tqdm
from training.evaluador import Evaluador
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f"./models/trained/{datetime.now().strftime("%Y%m%d_%H%M%S")}.log", level=logging.INFO)

with open("config.yaml") as f:
    logger.info('cargando config.yaml ...')
    params = yaml.safe_load(f)
    logger.info('config.yaml cargado ...')

ventana = params["poblacion"] // params["progenitores"]
params["n_inputs"] = (
    2 * params["pers_ovejas"]  # ovejas
    + 2 * params["pers_pastores"]  # otros pastores
    + 4  # el objetivo y su propia posicion
)
params["n_bits"] = 8 * (
    (
        params["n_inputs"] * params["hidden_lay_1"]
        + params[
            "hidden_lay_1"
        ]  # capa oculta 1: por cada neurona, un peso por input y un bias
    )
    + (
        params["hidden_lay_1"] * params["hidden_lay_2"]
        + params["hidden_lay_2"]  # capa oculta 2
    )
    + (params["hidden_lay_2"] * 2 + 2)  # capa de salida: 2 neuronas de salida, FIJO
)

rng: np.random.Generator = np.random.default_rng()


# inicializador de cada hilo
def worker_loop(in_q: mp.Queue, out_q: mp.Queue, params):
    local_rng: np.random.Generator = np.random.default_rng()
    ev = Evaluador(params, local_rng)
    for i, genome in iter(in_q.get, None):
        out_q.put((i, ev.evaluar(genome)))


def evaluar_poblacion(poblacion, in_q, out_q):
    fit = [0 for _ in range(len(poblacion))]
    for i, gen in enumerate(poblacion):
        in_q.put((i, gen))
    for _ in range(len(poblacion)):
        i, f = out_q.get()
        fit[i] = f
    return fit


# ====================================

# ======================= EVOLUCION =======================

if __name__ == "__main__":  # esto lo necesita multiprocessing para no joder
    logger.info('iniciando workers ...')
    N_WORKERS = 6  # OJO!
    # Arrancar workers
    in_q, out_q = mp.Queue(), mp.Queue()
    workers: list[mp.Process] = [
        mp.Process(target=worker_loop, args=(in_q, out_q, params))
        for _ in range(N_WORKERS)
    ]
    for w in workers:
        w.start()
    logger.info('workers inicializados ...')

    logger.info('inicializando poblacion ...')
    # 1) inicializar la poblacion al azar
    poblacion = rng.integers(
        0, 2, (params["poblacion"], params["n_bits"]), dtype=np.uint8
    )
    fit_history = np.empty((0, 2), dtype=float)
    logger.info('poblacion inicializada ...')

    # 2) calcular fitness
    fit = evaluar_poblacion(poblacion, in_q, out_q)
    sorted = np.argsort(fit)  # indices que ordenan de menor a mayor

    fit_elite = fit[sorted[-1]]
    fit_history = np.vstack([fit_history, [0, fit_elite]])
    logger.info('iniciando evolucion')
    t_ini = time.perf_counter()
    for g in tqdm(range(params["generaciones"])):
        # 1) elegir progenitores: un elite y el resto por ventana
        progenitores = np.empty(
            (params["progenitores"], params["n_bits"]), dtype=np.uint8
        )
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
            c1, c2 = np.sort(rng.integers(0, params["n_bits"], 2))

            # cruzar: segmento del padre 1 y el resto del padre 2
            poblacion[i, :c1] = progenitores[p2, :c1]  # antes de c1
            poblacion[i, c1:c2] = progenitores[p1, c1:c2]  # c1 a c2
            poblacion[i, c2:] = progenitores[p2, c2:]  # de c2 al final

            # mutar
            if rng.random() < params["mutacion"]:
                b = rng.integers(0, params["n_bits"])
                poblacion[i, b] ^= 1  # invierte 0 a 1 y viceversa

        # 3) evaluar fitness
        fit = evaluar_poblacion(poblacion, in_q, out_q)
        sorted = np.argsort(fit)  # indices que ordenan de menor a mayor
        if fit[sorted[-1]] > fit_elite:
            logger.info(f"generacion {g+1} - fitness superado: {fit_elite} -> {fit[sorted[-1]]}")
            fit_elite = fit[sorted[-1]]
            fit_history = np.vstack([fit_history, [g + 1, fit_elite]])

    t_total = time.perf_counter() - t_ini
    logger.info(f"evolucion terminada exitosamente")

    logger.info(f"deteniendo workers")
    # Detener workers
    for _ in range(N_WORKERS):
        in_q.put(None)
    for w in workers:
        w.join()

    print(f"Best fitness: {fit_elite}")
    print("fit history:")
    print(fit_history)

    logger.info(f"guardando modelo de mejor individuo")
    best_genome = poblacion[sorted[-1]]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "models/trained"
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, f"{timestamp}.npy"), best_genome)

    model_info = {
        "n_inputs": params["n_inputs"],
        "hidden_lay_1": params["hidden_lay_1"],
        "hidden_lay_2": params["hidden_lay_2"],
        "min_w": params["min_w"],
        "max_w": params["max_w"],
        "poblacion": params["poblacion"],
        "progenitores": params["progenitores"],
        "mutacion": params["mutacion"],
        "generaciones": params["generaciones"],
        "best_fitness": float(fit_elite),
        "tiempo": t_total,
    }

    with open(os.path.join(save_dir, f"{timestamp}.json"), "w") as f:
        json.dump(model_info, f, indent=2)

    logger.info(f"guardado modelo {timestamp}.json")

    logger.info(f"Programa finalizado en {t_total/60.0} minutos")