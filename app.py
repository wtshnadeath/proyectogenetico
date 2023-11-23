import time
from flask import Flask, request
from flask import render_template
from math import inf
from math import sqrt
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


app = Flask(__name__)

salon_de_la_fama = []

def crear_matriz(n_ciudades):
  matriz = [[0 for _ in range(n_ciudades)] for _ in range(n_ciudades)]
  for i in range(n_ciudades):
    for j in range(i + 1, n_ciudades):
      distancia = random.randint(1, 100)
      matriz[i][j] = distancia
      matriz[j][i] = distancia
  return matriz

def crear_poblacion(t_poblacion, n_ciudades):
  poblacion = []
  for _ in range(t_poblacion):
    individuo = random.sample(range(0, n_ciudades), n_ciudades)
    poblacion.append(individuo)
  return poblacion

def obtener_mejor_aptitud(poblacion, matriz):
  m_aptitud = float(inf)
  for individuo in poblacion:
    suma = 0
    for i in range(len(individuo) - 1):  # Recorrer hasta el penúltimo elemento
      ciudad_actual = individuo[i]
      siguiente_ciudad = individuo[i + 1]
      suma += matriz[ciudad_actual][siguiente_ciudad]
    if suma < m_aptitud:
      m_aptitud = suma
      chidito = individuo
      salon_de_la_fama.append(chidito)
  return m_aptitud

def obtener_peor_aptitud(poblacion, matriz):
  peor_aptitud = float(-inf)  # Usamos -inf para inicializar la peor aptitud
  for individuo in poblacion:
    suma = 0
    for i in range(len(individuo) - 1):
      ciudad_actual = individuo[i]
      siguiente_ciudad = individuo[i + 1]
      suma += matriz[ciudad_actual][siguiente_ciudad]
      if suma > peor_aptitud:
        peor_aptitud = suma
  return peor_aptitud

def obtener_promedio(poblacion,matriz):
  promedio = 0
  for individuo in poblacion:
    suma = 0
    for i in range(len(individuo) - 1):
      ciudad_actual = individuo[i]
      siguiente_ciudad = individuo[i + 1]
      suma += matriz[ciudad_actual][siguiente_ciudad]
    promedio += suma
  return round(promedio/len(poblacion),2)

def obtener_desviacion_estandar(poblacion,matriz,promedio):
  desviacion = 0
  for individuo in poblacion:
    suma = 0
    for i in range(len(individuo) - 1):
      ciudad_actual = individuo[i]
      siguiente_ciudad = individuo[i + 1]
      suma += matriz[ciudad_actual][siguiente_ciudad]
    distancia = (suma-promedio)**2
    desviacion += distancia
  return round(sqrt(desviacion/len(poblacion)),2)

def mutar(poblacion,p_mutacion):
  for ind in poblacion:
    if (random.uniform(0,1)<p_mutacion):
      mitad = len(ind) // 2
      ind[:mitad], ind[mitad:] = ind[mitad:], ind[:mitad]
  return poblacion

def cruzar(poblacion,p_cruza):
  nueva_generacion = []
  for i in range(0, len(poblacion), 2):
    if random.uniform(0, 1) < p_cruza:
      padre1 = poblacion[i]
      padre2 = poblacion[i + 1]
      punto_cruza = random.randint(1, len(padre1) - 1)  # Punto de cruza aleatorio
      hijo1 = padre1[:punto_cruza] + padre2[punto_cruza:]
      hijo2 = padre2[:punto_cruza] + padre1[punto_cruza:]
      nueva_generacion.extend([hijo1, hijo2])
    else:
      nueva_generacion.extend([poblacion[i], poblacion[i + 1]])
    return nueva_generacion
  
def plot_evolucion(info):
  matriz_np = np.array(info)
  nombres_variables = ['Generación', 'Mejor Aptitud', 'Peor Aptitud', 'Promedio', 'Desviación Estándar']
  for i in range(matriz_np.shape[1]):
    plt.plot(matriz_np[:, i][1:], label=nombres_variables[i])
  plt.xlabel('Tiempo')
  plt.ylabel('Valor de las Variables')
  plt.legend()
  plt.title('Valores a lo largo de las generaciones')
  timestamp = time.strftime('%Y%m%d%H%M%S')
  ruta_guardado = f'static/figura_{timestamp}.png'
  plt.savefig(ruta_guardado)
  return ruta_guardado


def plot_evolucion3(info):
  matriz_np = np.array(info)
  nombres_variables = ['Generación', 'Mejor Aptitud', 'Peor Aptitud', 'Promedio', 'Desviación Estándar']

  plt.figure(figsize=(10, 6))

  for i in range(1, matriz_np.shape[1]):
    sns.lineplot(x=matriz_np[:, 0], y=matriz_np[:, i], label=nombres_variables[i])

  plt.xlabel('Tiempo')
  plt.ylabel('Valor de las Variables')
  plt.title('Valores a lo largo de las generaciones')
  plt.legend()

  timestamp = time.strftime('%Y%m%d%H%M%S')
  ruta_guardado = f'static/figura_{timestamp}.png'
  plt.savefig(ruta_guardado)
  plt.close()
    
  return ruta_guardado

@app.route('/')
def index():
  return render_template('index.html')



@app.route('/procesar_formulario', methods=['POST'])
def procesar_formulario():
  matriz = []
  datos = []
  plt.pause(0.01)
  n_ciudades = 0
  t_poblacion = int(request.form['tam_pob'])
  p_cruza = float(request.form['prb_cru'])
  p_mutacion = float(request.form['prb_mut'])
  n_generaciones = int(request.form['num_gen'])
  file = request.files['arch_matriz']
  if file is not None and file.filename != '':
    matrix_content = file.read().decode('utf-8')
    matriz = np.array([list(map(float, row.split())) for row in matrix_content.splitlines()])
    n_ciudades = matriz.shape[1]
    poblacion = crear_poblacion(t_poblacion,n_ciudades)
  else:
    n_ciudades = int(request.form['tam_matriz'])
    matriz = crear_matriz(n_ciudades)
    poblacion = crear_poblacion(t_poblacion,n_ciudades)
  for i in range(n_generaciones):
    local = []
    mejor_aptitud = obtener_mejor_aptitud(poblacion,matriz)
    peor_aptitud = obtener_peor_aptitud(poblacion,matriz)
    promedio = obtener_promedio(poblacion,matriz)
    desviacion_estandar = obtener_desviacion_estandar(poblacion,matriz,promedio)
    local.append(i)
    local.append(mejor_aptitud)
    local.append(peor_aptitud)
    local.append(promedio)
    local.append(desviacion_estandar)
    poblacion = cruzar(poblacion,p_cruza)
    poblacion = mutar(poblacion,p_mutacion)
    datos.append(local)
  datos = sorted(datos, key=lambda x: x[1], reverse=True)
  ruta = plot_evolucion3(datos)
  return render_template('resultados.html',datos=datos, imagen_path = ruta)


if __name__ == '__main__':
  app.run(debug=True)

