import random

# Función de activación escalón
def funcion_escalon(x):
    return 1 if x >= 0 else 0

# Función de activación softmax aproximada sin usar exponencial
def funcion_softmax(salidas):
    max_salida = max(salidas)  # Encontrar la salida máxima para estabilizar el cálculo
    suma = sum([1 / (1 + (max_salida - s)) for s in salidas])  # Inversa de la diferencia
    return [1 / (1 + (max_salida - s)) / suma for s in salidas]  # Normalizar las salidas

# Inicialización de pesos y sesgos para una capa de manera aleatoria
def inicializar_pesos_capa(num_entradas, num_neuronas):
    pesos = [[random.uniform(-0.5, 0.5) for _ in range(num_entradas)] for _ in range(num_neuronas)]
    sesgos = [random.uniform(-0.5, 0.5) for _ in range(num_neuronas)]
    return pesos, sesgos

# Cálculo de la salida de una capa con función de activación escalón
def calcular_salida_capa_oculta(entradas, pesos, sesgos):
    salidas = []
    for i in range(len(pesos)):
        salida_lineal = sum(entradas[j] * pesos[i][j] for j in range(len(entradas))) + sesgos[i]
        salidas.append(funcion_escalon(salida_lineal))
    return salidas

# Cálculo de la salida de la capa de salida
def calcular_salida_capa_salida(entradas, pesos, sesgos):
    salidas = []
    for i in range(len(pesos)):
        salida_lineal = sum(entradas[j] * pesos[i][j] for j in range(len(entradas))) + sesgos[i]
        salidas.append(salida_lineal)
    # Aplicar la función softmax aproximada a las salidas lineales
    return funcion_softmax(salidas)

# Generar una imagen de movimiento lineal (línea diagonal)
def generar_imagen_lineal():
    imagen = [0] * 100
    for i in range(10):
        imagen[i * 11] = 1
    return imagen

# Generar una imagen de movimiento circular
def generar_imagen_circular():
    imagen = [0] * 100
    centro = 50  # Centro de la imagen 10x10
    radio = 3
    for i in range(10):
        for j in range(10):
            if (i - 5)**2 + (j - 5)**2 <= radio**2:
                imagen[i * 10 + j] = 1
    return imagen

# Mostrar la imagen en formato 10x10
def mostrar_imagen(imagen):
    for i in range(10):
        fila = imagen[i*10:(i+1)*10]
        print(" ".join(str(pixel) for pixel in fila))

# Generar conjunto de prueba
def generar_conjunto_prueba():
    ejemplos_lineales = [generar_imagen_lineal() for _ in range(30)]
    ejemplos_circulares = [generar_imagen_circular() for _ in range(30)]
    etiquetas_lineales = [0] * 30
    etiquetas_circulares = [1] * 30
    return ejemplos_lineales + ejemplos_circulares, etiquetas_lineales + etiquetas_circulares

# Ejemplo de uso con una capa oculta y una capa de salida
if __name__ == "__main__":
    # Generar conjunto de prueba
    ejemplos, etiquetas = generar_conjunto_prueba()

    # Parámetros de la red
    num_entradas = 100
    num_neuronas_capa_oculta = 10
    num_neuronas_capa_salida = 2  # Por ejemplo, dos categorías: línea y círculo

    # Inicializar pesos y sesgos para la capa oculta
    pesos_oculta, sesgos_oculta = inicializar_pesos_capa(num_entradas, num_neuronas_capa_oculta)

    # Inicializar pesos y sesgos para la capa de salida
    pesos_salida, sesgos_salida = inicializar_pesos_capa(num_neuronas_capa_oculta, num_neuronas_capa_salida)

    # Clasificar los ejemplos y calcular la precisión
    predicciones_correctas = 0
    for i, ejemplo in enumerate(ejemplos):
        # Calcular la salida de la capa oculta
        salida_capa_oculta = calcular_salida_capa_oculta(ejemplo, pesos_oculta, sesgos_oculta)

        # Calcular la salida de la capa de salida
        salida_capa_salida = calcular_salida_capa_salida(salida_capa_oculta, pesos_salida, sesgos_salida)

        # Determinar la predicción
        prediccion = 1 if salida_capa_salida[1] > salida_capa_salida[0] else 0

        # Comparar con la etiqueta conocida
        if prediccion == etiquetas[i]:
            predicciones_correctas += 1

    # Calcular la precisión
    precision = predicciones_correctas / len(ejemplos)
    print(f"Precisión del modelo: {precision * 100:.2f}%")
