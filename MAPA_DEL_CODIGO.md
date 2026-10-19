# Mapa del código: resComp

15/10/2026

El repositorio [lamascarlos/resComp](https://github.com/lamascarlos/resComp) contiene dos paquetes de Python (unas 23.600 líneas): `agnw_sim` simula la red de nanohilos de plata y `agnw_rc` la usa como reservoir computer.

## Cómo encaja todo

Una corrida recorre siempre la misma cadena. Los pasos 1–3 vienen de `agnw_sim` y los pasos 4–8, de `agnw_rc`.

1. **Geometría**: se tiran nanohilos al azar en un cuadrado y se detectan sus cruces, que son las uniones (`agnw_sim.geometry`, `agnw_rc.config.make_network`).
2. **Circuito**: cada hilo aporta su resistencia y cada cruce, una unión con conductancia variable. Se resuelve Kirchhoff con matrices ralas (`dualnode`, `circuit`, `cached_solver`; para varios electrodos, `agnw_rc.multicircuit`).
3. **Modelo de unión**: en cada paso, el voltaje sobre cada unión actualiza su estado (binario en `agnw_sim.dynamics`; promediado, filamento o volátil en `agnw_rc`).
4. **Entrada**: la señal u\[k\] se convierte en voltajes sobre los electrodos y cada valor se mantiene durante `hold` ms (`drive`, `multidrive`, `electrodes`).
5. **Observables**: se registran las corrientes de los electrodos, muestreadas varias veces por hold ("nodos virtuales") (`evaluate.observables`).
6. **Readout**: con ridge regression se entrena solo una combinación lineal de esas corrientes (`readout`).
7. **Tareas y métricas**: memoria (MC), no linealidad (IPC2), NARMA-10 y seno/cuadrada (`tasks`, `metrics`, `evaluate`).
8. **Veredicto**: se compara con referencias (`baselines`, incluida una ESN) y se deciden G0 (reproducible), G1 (computa más allá de la entrada) y G2 (≥ 50 % de una ESN) (`criteria`).

## agnw\_sim: el simulador físico

Es la carpeta `AgNW_FDT_Memristor_Framework/`, que contiene una copia reducida del commit `ffba009`. El proyecto de reservoir la importa sin modificarla.

| Módulo | Qué hace | Líneas |
| --- | --- | --- |
| `geometry.py` | Genera los nanohilos y detecta sus intersecciones | 156 |
| `dualnode.py` | Modelo de doble nodo (Lamas et al. 2026, Sec. 2.2): grafo del circuito con resistencia de cada hilo | 109 |
| `circuit.py` | Arma la matriz de admitancias rala y resuelve con condiciones de Dirichlet | 108 |
| `cached_solver.py` | Solver acelerado que reutiliza la factorización LU entre pasos | 199 |
| `dynamics.py` | Conmutación estocástica SET/RESET de cada unión | 99 |
| `simulate.py` | Bucle temporal: arma la red, avanza circuito + dinámica y guarda la trayectoria | 161 |
| `waveforms.py` | Formas de onda del voltaje en los electrodos | 30 |
| `presets.py` | Conjuntos de parámetros con nombre, p. ej. `lamas2026_fig4` | 24 |
| `io.py` | Guardado de trayectorias en HDF5 | 100 |
| `fdtest.py`, `fdtest_protocol.py`, `fdtest_observables.py` | Test de fluctuación-disipación (FDTest) para detectar memoria no markoviana | 570 + 235 + 131 |
| `markov_tests.py`, `exact_chain.py`, `benchmarks.py` | Diagnósticos de memoria y casos exactos para validar el FDTest | 214 + 61 + 74 |
| `experiment.py` | Lee mediciones exportadas del Keithley (Clarius+) para el FDTest | 114 |
| `reservoir.py` | Reservoir computing básico (no lo usa `agnw_rc`) | 475 |
| `analysis.py`, `visualize.py` | Autocorrelación, área de histéresis y gráficos de control | 73 + 48 |

Los `scripts/` de esta carpeta corren el FDTest y reproducen la Fig. 4 de Lamas et al. (`replicate_lamas_fig4.py`). La teoría está en `docs/switching_model.md` y `docs/theory_audit/`.

## agnw\_rc: el reservoir computing

Está en `AgNW_Reservoir_Computing_repo/agnw_rc/`. Los modelos de unión son lo que más cambia entre experimentos.

| Grupo | Módulo | Qué hace |
| --- | --- | --- |
| Configuración | `config.py` | Modelo de trabajo, `RunSpec` (los parámetros de una corrida) y las grillas de todos los experimentos |
| Modelo de unión | `meanfield.py` | Dinámica promediada sin ruido (N1/N2) |
| Modelo de unión | `filament.py`, `fig4.py` | Filamento continuo con ruido, calibrado con la Fig. 4 de Lamas et al. (N4) |
| Modelo de unión | `volatile.py` | Unión volátil basada en datos medidos, con rupturas (N10, la de los resultados principales) |
| Circuito y entrada | `multicircuit.py` | Kirchhoff con muchos electrodos sobre el grafo del framework |
| Circuito y entrada | `electrodes.py` | Geometrías de electrodos: bandas en los bordes, grilla de pads |
| Circuito y entrada | `drive.py` | Corrida con dos electrodos usando `run_simulation` del framework |
| Circuito y entrada | `multidrive.py` | Bucle temporal con muchos electrodos; codificación de la entrada y lecturas |
| Aprendizaje | `tasks.py` | Señales y tareas: memoria, IPC2, NARMA-10, seno/cuadrada |
| Aprendizaje | `readout.py` | Ridge regression por SVD, con alpha elegido por tarea |
| Evaluación | `metrics.py`, `evaluate.py` | NMSE, capacidad, consistencia entre réplicas, piso de ruido |
| Evaluación | `baselines.py` | Referencias sin nanohilos: línea de retardo y ESN |
| Evaluación | `criteria.py` | Punto de operación y veredictos G0/G1/G2 |
| Calibración N11 (en curso) | `strip.py`, `n11sim.py`, `series_stats.py`, `n11loss.py` | Tira rectangular como el dispositivo medido, protocolos de medición, estadística de series y función de pérdida contra Milano 2025 |
| Geometría N12 | `substrate.py` | Distribución de los hilos en el sustrato: clusters gaussianos en posiciones fijas y orientación normal (μ, σ), con N fijo; se elige con `RunSpec(geometry=...)`. Ver `docs/N12_GEOMETRY.md` |

## Scripts, tests y carpetas de apoyo

Cada experimento tiene su script en `scripts/`. Cada script escribe `results/*.json` y sus figuras.

| Carpeta / script | Para qué sirve |
| --- | --- |
| `scripts/run_sims.py` | Corridas en paralelo y reanudables de una grilla (`e1`, `n10_1`, `n10m`, ...) |
| `scripts/n10*_*.py` | Un script por experimento N10 (electrodos, ruido, rupturas, tamaño, layout) con subcomandos `sweep` / `run` / `report` |
| `scripts/n11_data.py`, `n11_fit.py` | Estadística de los datos de Milano 2025 y calibración N11 |
| `scripts/evaluate_all.py` | Recalcula todos los puntajes y genera `results/metrics.json` y `summary.csv` |
| `scripts/make_figures.py` | Todas las figuras de resultados (`--only`, `--mode paper\|presentation`) |
| `tests/` | Suite de pytest para la física, el solver, el readout, las métricas y los criterios |
| `cluster/` | Trabajos SLURM para el cluster CCT-Rosario |
| `results/` | Puntajes y veredictos (`*.json`, `summary.csv`). No incluye las corridas crudas (\~11 GB) |
| `figures/` | Figuras de los informes. `figures/README.md` dice qué script hace cada una |
| `data/milano2025/` | Datos medidos de Milano et al. 2025 (Nat. Commun., CC-BY) |
| `docs/` | Manual, resultados, diseños y reportes |

## Instalar y correr

Primero hay que instalar el framework, porque el proyecto de reservoir depende de él. Hace falta Python ≥ 3.10. Las dos carpetas tienen que quedar una al lado de la otra, y la del framework tiene que llamarse `AgNW_FDT_Memristor_Framework`.

```bash
cd AgNW_shared/AgNW_Reservoir_Computing_repo
python3 -m venv .venv
.venv/bin/pip install -e ../AgNW_FDT_Memristor_Framework
.venv/bin/pip install -e ".[dev]"          # numpy, scipy, matplotlib, pytest
.venv/bin/python -m pytest -q              # suite completa, ~7 min
```

Ejemplo de una corrida con la unión volátil y 16 pads. Con 800 muestras de 1 s cada una, tarda horas:

```python
from agnw_rc import config as C
from agnw_rc.multidrive import simulate_multi
from agnw_rc.evaluate import observables, score_iid, noise_floor

wo = C.model_washout_steps("volatile", tau_s=3.0, tau_spread=1.0)
spec = C.RunSpec(stream="iid", hold=1000, v_center=0.38, v_width=0.16,
                 network_seed=1, replicate=0, n_samples=800, model="volatile",
                 layout="pads", tau_s=3.0, tau_spread=1.0, washout=wo,
                 n_virtual=50, stride=2)
raw = simulate_multi(spec)
F = observables(raw)["electrodes"]
s = score_iid(F, raw["u"], floor=noise_floor(raw["u"]))
print(s["MC"], s["IPC2"], s["narma10_nmse"])
```

Los scripts que corren en paralelo fijan `OMP_NUM_THREADS=1`; conviene dejarlo así.

## Resultados y dónde seguir leyendo

Con uniones realistas, la red computa más allá de su entrada. Pero las rupturas, a la tasa medida, le quitan la mitad de la capacidad y hacen fallar a 1 de cada 4 dispositivos. Este es el resumen del 13/10/2026.

- **Uniones originales del framework**: no funcionan como reservoir (MC ≤ 0,26, al nivel de la entrada sola).
- **Unión volátil N10 con 16 pads**: MC 3,5, grado 2 6,7, NARMA-10 0,40. Cumple G1 y cumple G2 solo con margen ajustado.
- **Rupturas**: la red tolera unas 10⁻⁴ /s por unión, pero los datos de Milano 2025 implican de 0,005 a 0,03 /s.
- **Tamaño y layout** (1–4 mm, a 10⁻² /s): cumplen G1 alrededor de 3 de cada 4 redes, sin tendencia con el tamaño. Las bandas en los bordes son más robustas que los pads desde 2 mm.

Para profundizar, en este orden (rutas dentro de `AgNW_shared/AgNW_Reservoir_Computing_repo/`):

1. `docs/summary_2026-10-13/summary.md`: el resumen en una página.
2. `README.md`: visión general del proyecto y nombres de los experimentos.
3. `docs/RESULTS.md`: cada experimento con sus números y figuras.
4. `docs/MANUAL.md`: los modelos y el código, módulo por módulo.
5. `docs/N10_DESIGN.md`: el diseño de la unión volátil.
