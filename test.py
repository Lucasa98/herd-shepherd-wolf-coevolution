import sys
print(sys.path)
try:
    import site
    print("Cargado OK")
except ImportError:
    print("NO LO CARGA")
