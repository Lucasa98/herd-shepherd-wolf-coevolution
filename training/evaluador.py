import numpy as np
from models.NNShepherd import NNShepherdModel
from sim.world import World
from training.utils import Utils


class Evaluador:
    def __init__(self, params, rng):
        self.rng = rng
        self.params = params
        self.world: World = World(params["w_w"], params["w_h"], params, rng)
        self.w_cohesion = self.params["w_cohesion"]
        self.w_goal = self.params["w_goal"]
        self.w_inside = self.params["w_inside"]
        self.w_drive = self.params["w_drive"]
        self.w_finish = self.params["w_finish"]
        self.redundancia = self.params["redundancia"]

    def evaluar(self, gen: np.ndarray[np.uint8], N_steps: int) -> float:
        nn_model = Utils.genome_to_model(
            gen,
            self.params["n_inputs"],
            self.params["hidden_lay_1"],
            self.params["hidden_lay_2"],
            self.params["min_w"],
            self.params["max_w"],
        )
        shepherdModel = NNShepherdModel(self.params, self.rng, nn_model)

        cohesion_term = 0.0
        to_goal_progress_term = 0.0
        inside_term = 0.0
        driving_term = 0.0
        finish_term = 0.0

        # ===== redundancia =====
        for i in range(self.redundancia + 1):
            # ===== SIMULACION =====
            self.world.restart(shepherdModel)
            _, init_dist = self.world.cetroideYDistanciaCentroideObjetivo()

            c = 0
            while (
                c < N_steps
                and self.world.ticks_to_finish is None
                and not self.world.repitePosiciones()
            ):
                self.world.update()
                c += 1

            # ===== TERMINOS DEL FITNESS =====

            # (1) Cohesion: distancia de cada oveja al centroide
            cohesion_term += self.world.cohesionOvejas()

            # (2) progreso de la distancia inicial vs la distancia final al objetivo
            _, dist = self.world.cetroideYDistanciaCentroideObjetivo()
            to_goal_progress_term += max(0.0, (init_dist - dist) / init_dist)

            # (3) ovejas en el objetivo
            inside_term += self.world.ovejasDentroRate()  # already 0..1

            # (4) driving (0..1 scaled)
            driving_term += self.world.drivingRate()

            # (5) Si completa, bonus
            finish_term += 1.0 if self.world.ticks_to_finish is not None else 0.0

        cohesion_term *= self.w_cohesion / (self.redundancia + 1)
        to_goal_progress_term *= self.w_goal / (self.redundancia + 1)
        inside_term *= self.w_inside / (self.redundancia + 1)
        driving_term *= self.w_drive / (self.redundancia + 1)
        finish_term *= self.w_finish / (self.redundancia + 1)

        fitness = (
            cohesion_term
            + to_goal_progress_term
            + inside_term
            + driving_term
            + finish_term
        )

        return fitness, {
            "cohesion": cohesion_term,
            "to_goal": to_goal_progress_term,
            "inside": inside_term,
            "driving": driving_term,
            "finish": finish_term,
        }
