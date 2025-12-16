# Programacion en paradigmas funcional y paralelo
**INFO188 - Tarea 2: Multiplicacion de matrices en paralelo CPU y GPU**

---
## Integrantes
- Leonardo Moreno
- Elias Ojeda
- Marcelo Rojas
- Benjamin Uribe

---
## Descripcion
El objetivo de esta tarea es implementar y comparar el rendimiento de diferentes enfoques de programacion paralela para resolver el problema de la multiplicacion de matrices. Se implementaron las siguientes versiones:
1. **CPU multicore**
2. **GPU con CUDA**
3. **GPU con memoria compartida**
4. **GPU con tensor cores**

---
## Compilacion y Ejecucion

### Requisitos
- CUDA Toolkit instalado
- GPU compatible con CUDA


### Compilar
```bash
    make
```
### Ejecutar
```bash
    ./prog <n> <nt> <ALG>
```
**Parámetros:**
- `n`: Dimensión de las matrices (n x n)
- `nt`: Número de threads por bloque
- `ALG`: Algoritmo a utilizar
    - 1: CPU multicore
    - 2: GPU con CUDA
    - 3: GPU con memoria compartida
    - 4: GPU con tensor cores

**Ejemplos:**
```bash
    ./prog 1024 256 1    # Matrices 1024x1024, 256 threads/bloque, CPU multicore
    ./prog 1024 256 2    # Matrices 1024x1024, 256 threads/bloque, GPU con CUDA
    ./prog 1024 256 3    # Matrices 1024x1024, 256 threads/bloque, GPU con memoria compartida
    ./prog 1024 256 4    # Matrices 1024x1024, 256 threads/bloque, GPU con tensor cores
```
---

## Experimentos y Resultados

| N     | CPU (ms)  | GPU (ms)  | GPUsm (ms) | GPUtc (ms) |
|-------|-----------|-----------|------------|------------|
| 256   | 4.73862   | 2.33165   | 1.99469    | 1.77766    |
| 512   | 30.6007   | 2.72486   | 2.38285    | 1.9415     |
| 1024  | 222.825   | 7.7271    | 5.14045    | 2.67389    |
| 2048  | 1698.76   | 40.5996   | 31.5127    | 7.02566    |
| 4096  | 14559     | 277.462   | 200.308    | 41.7946    |
| 8192  | 215080    | 2305.56   | 1648.91    | 733.285    |
| 16384 | -         | 20043.7   | 13336.6    | 11214.1    |

### Rendimiento CPU con Diferentes Threads

| N    | Secuencial (ms) | 1 Thread (ms) | 2 Threads (ms) | 4 Threads (ms) | 8 Threads (ms) |
|------|-----------------|---------------|----------------|----------------|----------------|
| 256  | 100.432         | 20.773        | 10.64130       | 6.51839        | 4.81263        |
| 512  | 976.478         | 151.143       | 82.5704        | 43.4824        | 27.5622        |
| 1024 | 12862.672       | 1330.32       | 663.335        | 373.216        | 229.248        |

**Speedup respecto a versión secuencial:**

| N    | 1 Thread | 2 Threads | 4 Threads | 8 Threads |
|------|----------|-----------|-----------|-----------|
| 256  | 4.83x    | 9.44x     | 15.41x    | 20.86x    |
| 512  | 6.46x    | 11.83x    | 22.46x    | 35.44x    |
| 1024 | 9.67x    | 19.39x    | 34.47x    | 56.11x    |

---

## Especificaciones de Hardware

### CPU
- **Procesador**: [Intel core i7 11800H]
- **Núcleos/Hilos**: [8 cores / 16 threads]

### GPU
- **Modelo**: [NVIDIA GeForce RTX 3050 Ti Laptop GPU]
- **CUDA Cores**: [2560]
- **Tensor Cores**: [80]
- **Memoria**: [4 GB GDDR6]

### Sistema
- **RAM**: [16 GB DDR4 3200 MHz]
---

## Análisis de Resultados

### Comparación de Rendimiento

#### 1. Speedup respecto a CPU

| N     | GPU vs CPU | GPUsm vs CPU | GPUtc vs CPU |
|-------|-----------|--------------|--------------|
| 256   | 2.03x     | 2.38x        | 2.67x        |
| 512   | 11.23x    | 12.85x       | 15.77x       |
| 1024  | 28.84x    | 43.34x       | 83.34x       |
| 2048  | 41.84x    | 53.90x       | 241.78x      |
| 4096  | 52.47x    | 72.68x       | 348.30x      |
| 8192  | 93.30x    | 130.43x      | 293.30x      |

### Observaciones

#### Escalabilidad
- **CPU**: Muestra un crecimiento muy pronunciado en tiempo de ejecución. Para N=8192, el tiempo alcanza más de 215 segundos, siendo impráctica para matrices más grandes.
- **GPU básica**: Mejora significativa respecto a CPU, pero aún muestra limitaciones para matrices grandes.
- **GPU con memoria compartida**: Reduce considerablemente los tiempos al minimizar accesos a memoria global, mostrando mejoras de hasta 130x respecto a CPU.
- **GPU con tensor cores**: Ofrece el mejor rendimiento, especialmente notable en matrices grandes (N≥2048), alcanzando speedups de hasta 348x.

#### Comportamiento por Tamaño de Matriz

**Matrices pequeñas (N ≤ 512)**
- La diferencia entre implementaciones es menor
- El overhead de transferencia de datos a GPU reduce la ventaja
- Speedups modestos (2-15x)

**Matrices medianas (1024 ≤ N ≤ 4096)**
- Se observa un punto de inflexión donde las GPU muestran ventajas claras
- Los tensor cores comienzan a destacar significativamente
- Speedups de 28x a 348x

**Matrices grandes (N ≥ 8192)**
- Los tensor cores mantienen buen rendimiento
- GPU con memoria compartida muestra mejor escalabilidad que GPU básica
- CPU se vuelve completamente impráctico

---

## Interpretación y Conclusiones

### 1. Superioridad de GPU para Computación Intensiva
Las implementaciones en GPU superan consistentemente a la CPU multicore en todos los tamaños de matriz evaluados. Esto demuestra que problemas altamente paralelizables como la multiplicación de matrices se benefician enormemente de la arquitectura masivamente paralela de las GPU.

### 2. Importancia de la Optimización de Memoria
La versión con **memoria compartida** (GPUsm) muestra mejoras de 25-50% respecto a la implementación básica de GPU. Esto evidencia que:
- El acceso a memoria global es un cuello de botella significativo
- La reutilización de datos mediante memoria compartida reduce latencias
- La jerarquía de memoria en GPU debe aprovecharse para obtener rendimiento óptimo

### 3. Ventaja de Hardware Especializado
Los **tensor cores** (GPUtc) ofrecen el mejor rendimiento, especialmente para matrices grandes:
- Reducción de ~70% en tiempo respecto a GPU básica para N=4096
- Diseñados específicamente para operaciones matriciales

### 4. Trade-offs de Escalabilidad
- Para matrices pequeñas (N<512), el overhead de GPU puede no justificar su uso
- A partir de N=1024, las GPU muestran ventajas claras y crecientes
- La CPU se vuelve prohibitivamente lenta para N≥8192

### 5. Recomendaciones de Uso

**Usar CPU cuando:**
- Matrices muy pequeñas (N<256)
- No se dispone de GPU
- El overhead de transferencia de datos supera el beneficio computacional

**Usar GPU básica cuando:**
- Se dispone de GPU pero sin optimizaciones avanzadas
- Matrices de tamaño mediano (512≤N≤2048)

**Usar GPU con memoria compartida cuando:**
- Se requiere balance entre rendimiento y complejidad de implementación
- Matrices medianas a grandes (N≥1024)

**Usar tensor cores cuando:**
- Máximo rendimiento es crítico
- Matrices grandes (N≥2048)
- Se trabaja con operaciones matriciales frecuentes (deep learning, simulaciones)

### 6. Conclusión General
Este estudio demuestra que la elección del enfoque de paralelización debe considerar:
- **Tamaño del problema**: Matrices grandes favorecen GPU
- **Hardware disponible**: Aprovechar características específicas (tensor cores, memoria compartida)
- **Requisitos de rendimiento**: Balance entre tiempo de desarrollo y optimización

Los resultados confirman que la programación paralela en GPU, especialmente con optimizaciones específicas, es fundamental para computación de alto rendimiento.

---

## Notas Adicionales
- Todos los experimentos se realizaron bajo las mismas condiciones de hardware
- Para N=16384, no se pudo ejecutar la versión CPU debido a restricciones de tiempo/memoria
