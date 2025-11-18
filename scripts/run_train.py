import hidepygame
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

hidepygame


# inicializador de cada hilo
def worker_loop(in_q: mp.Queue, out_q: mp.Queue, params):
    local_rng: np.random.Generator = np.random.default_rng()
    ev = Evaluador(params, local_rng)
    for i, (genome, N_ticks) in iter(in_q.get, None):
        f, d = ev.evaluar(genome, N_ticks)
        out_q.put((i, f, d))


def evaluar_poblacion(poblacion, N_ticks: int, in_q, out_q):
    fit = [0 for _ in range(len(poblacion))]
    det = [0 for _ in range(len(poblacion))]
    for i, gen in enumerate(poblacion):
        in_q.put((i, (gen, N_ticks)))
    for _ in range(len(poblacion)):
        i, f, d = out_q.get()
        fit[i] = f
        det[i] = d
    return fit, det


if __name__ == "__main__":  # esto lo necesita multiprocessing para no joder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=f"./models/trained/{timestamp}.log", level=logging.INFO
    )

    with open("config.yaml") as f:
        logger.info("cargando config.yaml ...")
        params = yaml.safe_load(f)
        logger.info("config.yaml cargado ...")

    ventana = params["poblacion"] // params["progenitores"]
    params["n_inputs"] = (
        2 * params["pers_ovejas"]  # ovejas
        + 2 * params["pers_pastores"]  # otros pastores
        + 4  # el objetivo y su propia posicion
    )
    # capas ocultas: por cada neurona, un peso por input y un bias
    # capa de salida: 2 neuronas de salida, FIJO
    params["n_bits"] = 8 * (
        (params["n_inputs"] * params["hidden_lay_1"] + params["hidden_lay_1"])
        + (params["hidden_lay_1"] * params["hidden_lay_2"] + params["hidden_lay_2"])
        + (params["hidden_lay_2"] * 2 + 2)
    )

    # parametros de evolucion
    E = params["elite"]
    B = params["brecha_generacional"]
    N = params["poblacion"]
    P = params["progenitores"]
    n_bits = params["n_bits"]
    M = params["mutacion"]
    C = N - (E + B)  # numero de hijos a generar
    total_bits_children = C * n_bits  # total de bits entre hijos
    N_mut = int(M * total_bits_children)  # total de bits a mutar

    rng: np.random.Generator = np.random.default_rng()

    logger.info("iniciando workers ...")
    N_WORKERS = 7  # OJO!
    # Arrancar workers
    in_q, out_q = mp.Queue(), mp.Queue()
    workers: list[mp.Process] = [
        mp.Process(target=worker_loop, args=(in_q, out_q, params))
        for _ in range(N_WORKERS)
    ]
    for w in workers:
        w.start()
    logger.info("workers inicializados ...")

    # ======================= EVOLUCION =======================

    logger.info("inicializando poblacion ...")
    # 1) inicializar la poblacion al azar
    poblacion = rng.integers(
        0, 2, (N, params["n_bits"]), dtype=np.uint8
    )
    fit_history = np.empty((0, 2), dtype=float)
    logger.info("poblacion inicializada ...")

    # 2) calcular fitness
    N_steps = params["max_steps_ini"]
    steps_rate = (params["max_steps"] - params["max_steps_ini"]) / params[
        "generaciones"
    ]  # ratio de aumento del numero ticks
    fit, fit_detail = evaluar_poblacion(poblacion, N_steps, in_q, out_q)
    sorted = np.argsort(fit)  # indices que ordenan de menor a mayor

    fit_elite = fit[sorted[-1]]
    fit_history = np.vstack([fit_history, [0, fit_elite]])
    logger.info(
        "primer fitness: %.5f, %s",
        fit_elite,
        {k: f"{v:.5f}" for k, v in fit_detail[sorted[-1]].items()},
    )

    logger.info("iniciando evolucion")
    t_ini = time.perf_counter()
    for g in tqdm(range(P)):
        try:
            # 1) elegir progenitores: un elite y el resto por ventana
            progenitores = np.empty(
                (P, n_bits), dtype=np.uint8
            )

            v = ventana
            for i in range(P):
                progenitores[i] = poblacion[sorted[-rng.integers(0, v, 1)]]
                v += ventana

            # 2) reemplazo
            new_poblacion = poblacion.copy()

            # elite
            if E == 1:
                new_poblacion[0] = poblacion[sorted[-1]]

            # "supervivientes" (elite + brecha generacional)
            new_poblacion[E:E+B] = progenitores[:B]

            # hijos (cruza)
            i = E + B
            while i < N:
                p1, p2 = rng.integers(
                    0, P, 2
                )  # tomar progenitores al azar

                # punto de cruza: elegir inicio y fin del segmento a cruzar
                c1, c2 = np.sort(rng.integers(0, n_bits, 2))

                # cruzar generando dos individuos a la vez
                new_poblacion[i, :c1] = progenitores[p2, :c1]  # antes de c1
                new_poblacion[i, c1:c2] = progenitores[p1, c1:c2]  # c1 a c2
                new_poblacion[i, c2:] = progenitores[p2, c2:]  # de c2 al final
                i += 1
                if i < N:
                    new_poblacion[i, :c1] = progenitores[p1, :c1]  # antes de c1
                    new_poblacion[i, c1:c2] = progenitores[p2, c1:c2]  # c1 a c2
                    new_poblacion[i, c2:] = progenitores[p1, c2:]  # de c2 al final

            # mutar a nivel de gen
            if N_mut > 0:
                # Tomar indices aleatorios y flipear esos bits
                idx = rng.integers(0, total_bits_children, N_mut, dtype=np.int64)
                flat = new_poblacion[E+B:N].reshape(-1)
                flat[idx] ^= 1

            # ovejas y ticks para esta generacion
            N_steps += steps_rate
            # 3) evaluar fitness
            fit, fit_detail = evaluar_poblacion(
                poblacion, int(np.floor(N_steps)), in_q, out_q
            )
            sorted = np.argsort(fit)  # indices que ordenan de menor a mayor
            if fit[sorted[-1]] > fit_elite:
                logger.info(
                    f"generacion {g+1} - fitness superado: %.5f, %s",
                    fit[sorted[-1]],
                    {k: f"{v:.5f}" for k, v in fit_detail[sorted[-1]].items()},
                )
                fit_elite = fit[sorted[-1]]
                fit_history = np.vstack([fit_history, [g + 1, fit_elite]])
        except KeyboardInterrupt:
            logger.info(f"evolucion interrumpida en generacion {g}")
            break

    t_total = time.perf_counter() - t_ini

    if N_steps >= params["max_steps"]:
        logger.info("evolucion terminada exitosamente")

    logger.info("deteniendo workers")
    # Detener workers
    for _ in range(N_WORKERS):
        in_q.put(None)
    for w in workers:
        w.join()

    print(f"Best fitness: {fit_elite}")
    print("fit history:")
    print(fit_history)

    logger.info("guardando modelo de mejor individuo")
    best_genome = poblacion[sorted[-1]]
    save_dir = "models/trained"
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, f"{timestamp}-model.npy"), best_genome)

    with open(os.path.join(save_dir, f"{timestamp}-config.json"), "w") as f:
        json.dump(params, f, indent=2)

    logger.info(f"guardado modelo {timestamp}.npy")

    logger.info(f"Programa finalizado en {t_total/60.0} minutos")
