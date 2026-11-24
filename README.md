# Predicción de fallas y RUL industrial con MLOps

Prototipo educativo de vida útil remanente (**RUL**, en ciclos) con Random Forest, XGBoost y LSTM (PyTorch), MLflow, Weights & Biases opcional, PostgreSQL, FastAPI y Streamlit. Una alerta se activa cuando el RUL estimado **es menor o igual al umbral configurable** (30 ciclos por defecto).

**No es una probabilidad de falla ni un sistema validado para decisiones reales de mantenimiento o seguridad.** C-MAPSS contiene simulaciones de motores aeronáuticos, no mediciones de máquinas reales; los ciclos no equivalen automáticamente a horas.

## Datos y evaluación

Descarga el conjunto turbofan C-MAPSS desde el [repositorio de datos de NASA PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/), revisando sus condiciones de acceso y uso en la fuente. Extrae `train_FD001.txt`, `test_FD001.txt` y `RUL_FD001.txt` en `data/CMAPSSData/`. No se distribuyen aquí. La primera versión solo admite FD001; otros subconjuntos requieren evaluación y adaptación.

Cada fila tiene unidad, ciclo, tres ajustes operativos y 21 sensores. Se validan formato, valores finitos, ciclos positivos y consecutivos y ausencia de duplicados. Las unidades físicas y significado de cada sensor se deben consultar en la documentación original. `python -m rul.eda` genera estadísticas, distribución de duraciones y gráfico de sensores en `artifacts/eda/`.

El RUL de entrenamiento es el último ciclo observado menos el ciclo actual. Se separan motores antes de ajustar el escalador. La validación utiliza un prefijo censurado reproducible por motor y su RUL verdadero; la prueba oficial usa la última observación de cada motor y las etiquetas publicadas. El modelo se elige por RMSE de validación **antes** de leer las etiquetas de prueba. Repetir y ajustar usando el test oficial invalida esa garantía.

Todos los candidatos usan ventanas causales idénticas, normalizadas con estadísticas de entrenamiento; cuando falta historial, se repite la primera lectura. Random Forest, XGBoost y la línea base usan la ventana aplanada y LSTM la secuencia completa. Las predicciones negativas se recortan a cero. Se registran RMSE, MAE, puntuación asimétrica tipo NASA, MAE por rango y matriz de confusión, precisión y recall de la alerta. MLflow guarda parámetros, particiones, hashes SHA-256, métricas, predicciones y gráfico de validación, preprocesador y modelo; se asigna `champion` a la versión seleccionada.

## Inicio con Docker Compose

Requiere Docker Compose y conexión para construir la imagen y obtener los datos. Desde este directorio:

```bash
cp .env.example .env
# Coloca los tres archivos oficiales en data/CMAPSSData/ antes de entrenar.
docker compose build api
docker compose up -d --no-build postgres mlflow
docker compose run --rm trainer
docker compose up -d --no-build api dashboard
```

Todos los servicios comparten una imagen. Si el índice oficial de PyTorch redirige a un dominio inaccesible, sustituye el paso de construcción por `docker compose build --build-arg TORCH_WHEEL_URL=URL api`, usando la URL oficial de una rueda CPU compatible con Python 3.11 y la arquitectura de la máquina. Incluye su hash SHA-256 en la URL para verificarla.

Consulta MLflow en <http://localhost:5000>, la API en <http://localhost:8000/docs> y Streamlit en <http://localhost:8501>. El entrenamiento corre en CPU y puede tardar; `docker compose run --rm trainer --epochs 1` permite una prueba más breve, no una comparación definitiva. Para entrenar un subconjunto, usa por ejemplo `--models random_forest xgboost`. `docker compose logs -f` muestra errores; el servicio `/ready` devuelve 503 hasta que exista un modelo registrado y accesible.

PostgreSQL y artefactos MLflow utilizan volúmenes persistentes; `docker compose down` detiene los servicios sin eliminarlos. Las credenciales de `.env.example`, puertos publicados y falta de autenticación son **solo para uso local**. Configura acceso, secretos, TLS y controles apropiados antes de cualquier despliegue. No publiques `.env`, datos ni artefactos en Git.

W&B está desactivado por defecto y MLflow sigue funcionando sin conexión a W&B. Para habilitarlo, añade `WANDB_API_KEY` a `.env` y ejecuta `docker compose run --rm trainer --wandb --wandb-project TU_PROYECTO`; también necesitas conectividad y autorización. Un fallo con W&B activado interrumpe el entrenamiento de forma explícita. No incluyas la clave en comandos, logs o código.

## Ejecución local y pruebas

Con Python 3.11+ y una instalación de PyTorch adecuada a tu plataforma:

```bash
python -m venv .venv
. .venv/bin/activate
pip install 'torch>=2.2,<3' --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[dev]'
# Opcional: pip install -e '.[wandb]'
python -m rul.eda
python -m rul.train --tracking-uri http://localhost:5000
pytest -q
```

MLflow debe tener un backend compatible con registro de modelos, como el PostgreSQL de Compose; también puede usarse un backend SQLite para experimentos locales. Para usar la API fuera de Compose, define `DATABASE_URL` y `MLFLOW_TRACKING_URI` y ejecuta `uvicorn rul.api:app --port 8000`; ejecuta el dashboard con `streamlit run dashboard.py`. `MODEL_NAME`, `MODEL_ALIAS` y `RUL_ALERT_THRESHOLD` deben coincidir entre entrenamiento y servicio. Las pruebas usan datos sintéticos, no necesitan W&B ni GPU.

## API y dashboard

- `GET /health`: proceso activo; `GET /ready`: base de datos y modelo disponibles.
- `GET /model`: nombre, alias, versión y umbral del modelo.
- `POST /predict`: predice y guarda en una transacción la ventana enviada y el resultado. Cada lectura tiene `cycle`, una lista `settings` de 3 valores y una lista `sensors` de 21 valores. Los ciclos deben ser consecutivos y ascendentes; una lectura es suficiente y se rellena con la primera. El modelo toma como máximo las últimas 30 lecturas por defecto; la longitud real se guarda con el modelo.
- `GET /units` y `GET /history/{unit}`: unidades e historial persistente. El historial conserva versión y umbral usados en cada predicción.

Ejemplo mínimo, tras iniciar API y modelo:

```bash
python - <<'PY' | curl -sS -X POST http://localhost:8000/predict -H 'Content-Type: application/json' --data-binary @-
import json
print(json.dumps({"unit": "motor-1", "readings": [
    {"cycle": 1, "settings": [0.0] * 3, "sensors": [0.0] * 21}
]}))
PY
```

Los valores de ejemplo no constituyen una lectura industrial real. El dashboard acepta CSV con encabezados en este orden: `unit,cycle,setting_1,setting_2,setting_3,sensor_1,...,sensor_21`. Para generar uno a partir de un archivo oficial descargado:

```bash
python -c "from pathlib import Path; from rul.data import load_cycles; load_cycles(Path('data/CMAPSSData/test_FD001.txt')).to_csv('example.csv', index=False)"
```

El dashboard envía lecturas a la API, permite seleccionar el motor y muestra las predicciones persistidas, la evolución de sensores y la alerta. Si falta el modelo o la base de datos, no se devuelve un resultado simulado.

