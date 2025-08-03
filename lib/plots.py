import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def point_in_octhaedron(r):
  x, y, z = r
  return abs(x) + abs(y) + abs(z) <= 1

def point_in_polygon(r, polygon):
  rx, ry = r
  n = len(polygon)
  inside = False

  # Itera su ogni lato del poligono
  for i in range(n):
    x1, y1 = polygon[i]
    x2, y2 = polygon[(i + 1) % n]  # Collegamento all'elemento successivo, con wrap-around

    # Verifica se il raggio interseca il lato
    if ((y1 > ry) != (y2 > ry)) and (rx < (x2 - x1) * (ry - y1) / (y2 - y1) + x1):
      inside = not inside  # Inverti lo stato di "inside"
  return inside


def heavysideStepFunction(x):
  if x >= 0:
    return 1
  else:
    return 0

def f_N(x, alpha):
  return x*np.cos(alpha) + np.sqrt(1 - x**2)*np.sin(alpha)

def F4_POVM_evaluation(r, piano='XZ'):

  N = 4
  alpha = np.pi/N

  # Definire i vertici e calcolare il vettore r
  angles = [0, np.pi/2, np.pi, 3*np.pi/2]
  vertices = np.array([[np.cos(a), np.sin(a)] for a in angles])

  if piano == 'XZ':
    r1, r2 = r[0], r[2]
  elif piano == 'XY':
    r1, r2 = r[2], r[1]
  elif piano == 'YZ':
    r1, r2 = r[0], r[1]

  # Creare il grafico
  fig, ax = plt.subplots()
  circle = Circle((0, 0), radius=1, edgecolor='black', facecolor='none', linewidth=2)
  ax.add_patch(circle)

  # Disegnare i vertici e il punto r
  vertices_x, vertices_z = vertices[:, 0], vertices[:, 1]
  ax.plot(np.append(vertices_z, vertices_z[0]), np.append(vertices_x, vertices_x[0]), 'b-', linewidth=2)
  ax.scatter(r1, r2, color='red', marker='o', s=50)

  # Configurazione degli assi
  ax.set_xlim(-1.1, 1.1)
  ax.set_ylim(-1.1, 1.1)
  ax.grid(True)
  ax.set_aspect('equal')

  # Axes tags
  if piano == 'XZ':
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
  elif piano == 'XY':
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
  elif piano == 'YZ':
    ax.set_xlabel('Z')
    ax.set_ylabel('Y')

  # Mostrare il grafico
  plt.show()

  ak = [[np.cos(angle), np.sin(angle)] for angle in angles]
  ak = np.array(ak)
  uk = [(ak[i+1]-ak[i]) for i in range(len(ak)-1)]
  uk.append(ak[0]-ak[-1])  # Aggiungi il termine mancante
  uk = np.array(uk)  # Converte uk in un array NumPy
  uk = (1/np.sqrt(2)) * uk
  inside_polygon = point_in_polygon((r1, r2), vertices)

  if inside_polygon:
      print(f"Point in polygon: {inside_polygon}")
      guessing_prob = 2/N
      print(f'Guessing Probability: {guessing_prob}')
      print(f'Min-entropy: {-np.log2(guessing_prob)}')
  else:
      print(f"Point in polygon: {inside_polygon}")
      dot_products = [np.dot([r1, r2], uk[i]) for i in range(len(uk))]
      heavyside_steps = [heavysideStepFunction(dot_products[i] - np.cos(alpha)) for i in range(len(uk))]

      guessing_prob = 1 / N + (1 / N) * sum(f_N(dot_products[i], alpha) * heavyside_steps[i] for i in range(len(uk)))
      print(f'Guessing Probability: {guessing_prob}')
      print(f'Min-entropy: {-np.log2(guessing_prob)}')

  return inside_polygon, guessing_prob

def F6_POVM_evaluation(r):

  N = 6
  alpha = np.arccos(1/np.sqrt(3))

  # Generazione della sfera
  radius = 1
  phi = np.linspace(0, np.pi, 100)
  theta = np.linspace(0, 2 * np.pi, 100)
  phi, theta = np.meshgrid(phi, theta)
  x = radius * np.sin(phi) * np.cos(theta)
  y = radius * np.sin(phi) * np.sin(theta)
  z = radius * np.cos(phi)

  # Plot
  fig = plt.figure(figsize=(8, 8))
  ax = fig.add_subplot(111, projection='3d')


  # Define ochtaedron vertices
  ottaedro_vertici = [
      (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)
  ]

  # Spigoli dell'ottaedro
  ottaedro_spigoli = [
      (0, 4), (0, 2), (0, 3), (0, 5),  # Collegamenti dal vertice (1, 0, 0)
      (1, 4), (1, 2), (1, 3), (1, 5),  # Collegamenti dal vertice (-1, 0, 0)
      (2, 4), (2, 5),  # Collegamenti dal vertice (0, 1, 0)
      (3, 4), (3, 5)   # Collegamenti dal vertice (0, -1, 0)
  ]

  # Disegno dell'ottaedro
  for edge in ottaedro_spigoli:
      x_edge = [ottaedro_vertici[edge[0]][0], ottaedro_vertici[edge[1]][0]]
      y_edge = [ottaedro_vertici[edge[0]][1], ottaedro_vertici[edge[1]][1]]
      z_edge = [ottaedro_vertici[edge[0]][2], ottaedro_vertici[edge[1]][2]]
      ax.plot(x_edge, y_edge, z_edge, color='blue', linewidth=2)

  # Sphere
  ax.plot_surface(x, y, z, cmap='viridis', alpha=0.4, rstride=5, cstride=5, edgecolor='gray')

  # Point r
  ax.scatter(r[0], r[2], r[1], color='red', marker='o', s=50, label='Punto r')

  # Disegno dei vertici dell'ottaedro
#   ottaedro_x, ottaedro_y, ottaedro_z = zip(*ottaedro_vertici)
#   ax.scatter(ottaedro_x, ottaedro_y, ottaedro_z, color='blue', s=50, label='Vertici Ottaedro')

  # Configurazione degli assi con proporzioni uniformi
  ax.set_xlim([-1.2, 1.2])
  ax.set_ylim([-1.2, 1.2])
  ax.set_zlim([-1.2, 1.2])
  ax.set_box_aspect([1, 1, 1])  # Proporzioni uniformi

  # Etichettatura degli assi
  ax.set_xlabel('X')
  ax.set_ylabel('Z')
  ax.set_zlabel('Y')

  # Mostrare il grafico
  plt.show()

  # Defining all the versors for octhaedron faces
  uk = np.array([
      (1,1,1),
      (-1,1,1),
      (1,-1,1),
      (-1,-1,1),
      (1,1,-1),
      (-1,1,-1),
      (1,-1,-1),
      (-1,-1,-1)
  ])

  norms = np.linalg.norm(uk, axis=1, keepdims=True)
  uk_normalized = uk / norms

  # Guessing Probability in case r is in the octhaedron
  if point_in_octhaedron(r):
    guessing_prob = 2/N

  # Guessing Probability in case r is outside the octhaedron
  else:
    dot_products = []
    for u in uk_normalized:
      dot_products.append(np.dot(np.array(r), np.array(u)))

    heavyside_steps = [heavysideStepFunction(dot_products[i] - np.cos(alpha)) for i in range(len(uk))]
    sum = 0

    if heavyside_steps.count(1) == 1:
      for i in range(len(dot_products)):
        sum += f_N(dot_products[i], alpha) * heavyside_steps[i]
      # applying paper's equation 7
      guessing_prob = (1 / N) + ((1 / N) * sum)

    else:
      for i in range(len(dot_products)):
        vector = [f_N(dot_products[i], alpha) * heavyside_steps[i] for i in range(len(dot_products))]
        guessing_prob = 1 / N + (1 / N) * max(vector)

  print(f'Point is in octhaedron: {point_in_octhaedron(r)}')
  print(f'Guessing Probability: {guessing_prob}')
  print(f'Min-entropy:  {-np.log2(guessing_prob)}')
  print('======================================================================)')