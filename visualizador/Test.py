import numpy as np
from Visualizador import Visualizador

if __name__ == "__main__":
    vis = Visualizador(num_shepherds=1, num_sheeps=1, world_size=10)

    r = 3
    centro = np.array([5, 5])
    angulo = 0

    while True:
        x = centro[0] + r * np.cos(angulo)
        y = centro[1] + r * np.sin(angulo)
        pos = np.array([[x, y]])

        vis.render(shepherds=pos, sheeps=[[1,1]], goal=np.array([3, 5]))

        angulo += 0.5 * vis.deltatime()  # velocidad angular
        if angulo > 2 * np.pi:
            angulo -= 2 * np.pi
