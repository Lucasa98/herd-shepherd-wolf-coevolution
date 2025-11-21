# Coevolución de sistema multiagente para comportamiento emergente rebaño-pastor

Trabajo Final Creativo de Inteligencia Computacion (2025, FICH-UNL)

---

## Instalacion

1. Instalar [pipx](https://github.com/pypa/pipx?tab=readme-ov-file#on-windows)
2. Instalar [Poetry](https://python-poetry.org/docs/#installing-with-pipx)
3. Ejecutar `poetry install`.
4. Instalar el comando `poetry shell`: `poetry self add poetry-plugin-shell`

## Uso

- Ejecutar los scripts **SIEMPRE DESDE EL ENTORNO VIRTUAL**.
- Activar el entorno con `poetry shell` o ejecutar codigo directo con `poetry run python -m scripts.run_sim`.
- Para entrenar un modelo, usar `scripts/train_model.py`.
- Para correr la simulacion (con visualizador), usar `scripts/run_sim.py`.
# Referencia

- [Docs PyTorch](https://docs.pytorch.org/docs/stable/index.html)
- [Docs Pygame](https://www.pygame.org/docs/)
