# Modelo de Ising en 2-D usando CUDA

Este proyecto de CUDA implementa el algoritmo de Metropolis para el modelo de Ising en 2-dimensiones.

Este modelo consiste de $N = L^2$ espines que toman valores $s_i=\pm1$, colocados en una red cuadrada de dimensiones $L\times L$.
La energía de una configuración de este sistema es:

$$E = -\sum_{(ij)} s_is_j$$

donde $(ij)$ recore los pares de espines vecinos en la red.

En una simulación de Metrópolis de la dinámica de este sistema, un espín $i$ cambia de signo con una probabilidad

$$\min(1, \exp(-\beta\Delta E))$$

donde $\Delta E$ es la variación energética asociada al cambio de signo,

$$\Delta E = 2\sum_{j\in \mathcal N(i)} s_i s_j$$

y $\beta=1/T$ is el inverso de la temperatura.
(Para simplificar la notación usamos unidades donde la constante de Boltzmann es 1).

Se define la magnetización de una configuración de espines como:

$$m = \frac{1}{N}\sum_i s_i$$

Este sistema fue resuelto analíticamente por Onsager, en el límite termodinámico $N\rightarrow\infty$.
Onsager demostró que este sistema exhibe una transición de fase a la temperatura crítica

$$T_c = \frac{2}{\log(1 + \sqrt 2)} \approx 2.269185$$

o $\beta_c\approx0.440687$.
Para $\beta<\beta_c$, la magnetización es cero.
Para $\beta>\beta_c$, la magnetización es distinta de cero, y tiende a 1 a medida que $\beta$ crece.
Analíticamente, Onsager encontró que la magnetización para $\beta>\beta_c$ viene dada por:

$$m = [1-\sinh^{-4}(2\beta)]^{1/8}$$

Programamos una simulación Monte Carlo de este sistema en CUDA (fichero `ising.cu`), con $N=1024$ espines ($L=32$).
El comportamiento se grafica en la figura siguiente, que muestra la solución analítica y la magnetización encontrada en las simulaciones.

![Results](https://i.ibb.co/gwvVBJm/Ising.png "Results")

La discrepancia entre la solución analítica y las simulaciones se debe a que la solución de Onsager asume que el sistema es infinito ($N\rightarrow\infty$) mientras que nuestras simulaciones se realizan necesariamente en un sistema finito ($N=1024$).

Para reproducir estos resultados, compilamos `ising.cu` y lo ejectuamos con los siguientes comandos:

```
nvcc ising.cu -o ising --library=curand
./ising > ising.txt 
```

A partir de estos datos, la figura fue obtenida con el Notebook de Mathematica `plot.nb`.
