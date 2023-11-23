import random
import numpy as np

def crear_matriz(n_ciudades):
  matriz = [[0 for _ in range(n_ciudades)] for _ in range(n_ciudades)]
  for i in range(n_ciudades):
    for j in range(i + 1, n_ciudades):
      distancia = random.randint(1, 100)
      matriz[i][j] = distancia
      matriz[j][i] = distancia
  return matriz

crear_matriz(30)

matriz_generada = crear_matriz(30)

# Guardar la matriz en un archivo de texto
np.savetxt('matriz_distancias.txt', matriz_generada, fmt='%d', delimiter='\t')

print("Matriz generada y guardada en 'matriz_distancias.txt'")

