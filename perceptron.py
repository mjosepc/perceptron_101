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

# Mostrar la imagen en formato 10x10
def mostrar_imagen(imagen):
    for i in range(10):
        fila = imagen[i*10:(i+1)*10]
        print(" ".join(str(pixel) for pixel in fila))

# Ejemplo de uso con una capa oculta y una capa de salida
if __name__ == "__main__":
    # Generar una imagen aleatoria de 10x10 píxeles, aplanada a un vector de 100 elementos
    entradas = [random.choice([0, 1]) for _ in range(100)]

    # Mostrar la imagen generada
    print("Imagen de entrada:")
    mostrar_imagen(entradas)

    # Parámetros de la red
    num_entradas = 100
    num_neuronas_capa_oculta = 10
    num_neuronas_capa_salida = 2  # Por ejemplo, dos categorías: línea y círculo

    # Inicializar pesos y sesgos para la capa oculta
    pesos_oculta, sesgos_oculta = inicializar_pesos_capa(num_entradas, num_neuronas_capa_oculta)

    # Calcular la salida de la capa oculta
    salida_capa_oculta = calcular_salida_capa_oculta(entradas, pesos_oculta, sesgos_oculta)

    # Inicializar pesos y sesgos para la capa de salida
    pesos_salida, sesgos_salida = inicializar_pesos_capa(num_neuronas_capa_oculta, num_neuronas_capa_salida)

    # Calcular la salida de la capa de salida
    salida_capa_salida = calcular_salida_capa_salida(salida_capa_oculta, pesos_salida, sesgos_salida)

    # Mostrar las probabilidades de pertenencia a cada categoría
    print(f"\nProbabilidades de pertenencia: {salida_capa_salida}")
    print(f"La predicción es: {'círculo' if salida_capa_salida[1] > salida_capa_salida[0] else 'línea'}")
