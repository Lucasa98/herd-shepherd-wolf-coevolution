import numpy as np
from models.NNShepherd import NNShepherdModel
from sim.world import World
from training.utils import Utils


class Evaluador:
    def __init__(self, params, rng):
        self.rng = rng
        self.params = params
        self.world: World = World(params["w_w"], params["w_h"], params, rng)

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

        self.world.restart(shepherdModel)

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
        cohesion_term = self.world.cohesionOvejas()

        # (2) distancia del centroide al objetivo
        to_goal_term = self.world.distanciaCentroideObjetivo()

        # (3) ovejas en el objetivo
        inside_term = self.world.ovejasDentroRate()  # already 0..1

        # (4) driving (0..1 scaled)
        driving_term = self.world.drivingRate()

        # (5) Si completa, bonus
        finish_term = 1.0 if self.world.ticks_to_finish is not None else 0.0

        # ===== PESOS =====
        w_cohesion = 1
        w_goal = 0.5
        w_inside = 1.0
        w_drive = 3.0
        w_finish = 0.5  # si es alto, cuando uno gana de casualidad, el resto la tiene muy dificil

        cohesion_term *= w_cohesion
        to_goal_term *= w_goal
        inside_term *= w_inside
        driving_term *= w_drive
        finish_term *= w_finish

        fitness = (
            cohesion_term + to_goal_term + inside_term + driving_term + finish_term
        )

        return fitness, {
            "cohesion": cohesion_term,
            "to_goal": to_goal_term,
            "inside": inside_term,
            "driving": driving_term,
            "finish": finish_term,
        }
