# Predicción de fallas y vida útil remanente

Este proyecto estima la vida útil remanente (**RUL**, en ciclos) de motores turbofán usando el conjunto de datos C-MAPSS. Compara modelos Random Forest, XGBoost y LSTM, y reúne el flujo de trabajo con MLflow, FastAPI, Streamlit y PostgreSQL. Weights & Biases (W&B) es opcional.

Se genera una alerta cuando el RUL estimado es menor o igual al umbral configurado (30 ciclos por defecto).

> **Alcance:** es un prototipo educativo, no una herramienta validada para mantenimiento ni para decisiones de seguridad. C-MAPSS contiene simulaciones de motores aeronáuticos, no mediciones de máquinas reales; los ciclos tampoco equivalen automáticamente a horas.

## Datos y evaluación

El conjunto C-MAPSS está disponible en el [repositorio de datos de NASA PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/); las condiciones de acceso y uso se describen en la fuente. Los archivos `train_FD001.txt`, `test_FD001.txt` y `RUL_FD001.txt` deben ubicarse en `data/CMAPSSData/`. Los datos no se incluyen en este repositorio. Por ahora solo se admite FD001; los demás subconjuntos requieren adaptación y evaluación.

Cada registro contiene una unidad, su ciclo, tres ajustes operativos y 21 sensores. El cargador valida el formato, los valores finitos, los ciclos positivos y consecutivos y la ausencia de duplicados. Las unidades y el significado de los sensores se describen en la documentación original. El módulo `rul.eda` genera estadísticas, distribuciones y gráficos de sensores en `artifacts/eda/`:

```bash
python -m rul.eda
```

Los motores se separan antes de ajustar el escalador. El RUL de entrenamiento se calcula a partir del último ciclo observado de cada motor. La validación usa un prefijo censurado reproducible y su RUL real; la prueba oficial usa la última observación de cada motor y las etiquetas publicadas. El modelo se selecciona con el RMSE de validación antes de consultar las etiquetas de prueba. Ajustar el modelo con esas etiquetas invalida la evaluación.

Todos los candidatos usan las mismas ventanas causales, normalizadas con estadísticas de entrenamiento; si falta historial, se repite la primera lectura. Random Forest, XGBoost y la línea base reciben la ventana aplanada; LSTM procesa la secuencia completa. Las predicciones negativas se recortan a cero.

MLflow registra parámetros, particiones, hashes SHA-256, métricas, predicciones, gráficos de validación, preprocesador y modelo. La versión seleccionada recibe el alias `champion`. Además del RMSE y MAE, se calculan la puntuación asimétrica tipo NASA, el MAE por rango y las métricas de precisión y recall de la alerta.

## Inicio con Docker Compose

El flujo de Docker Compose requiere Docker Compose, conexión a internet para construir la imagen y los archivos C-MAPSS indicados anteriormente para entrenar:

```bash
cp .env.example .env
# Los tres archivos oficiales deben estar en data/CMAPSSData/.
docker compose build api
docker compose up -d --no-build postgres mlflow
docker compose run --rm trainer
docker compose up -d --no-build api dashboard
```

Los servicios comparten una imagen. Si el índice oficial de PyTorch redirige a un dominio inaccesible, la imagen puede construirse con una rueda CPU compatible con Python 3.11 y la arquitectura de la máquina:

```bash
docker compose build --build-arg TORCH_WHEEL_URL=URL api
```

La URL debe corresponder a una rueda oficial e incluir su hash SHA-256 para verificarla.

Al iniciar los servicios, MLflow está disponible en <http://localhost:5000>, la API en <http://localhost:8000/docs> y Streamlit en <http://localhost:8501>. El entrenamiento se ejecuta en CPU y puede tardar. Para una prueba rápida (no una comparación definitiva), se puede ejecutar `docker compose run --rm trainer --epochs 1`. También es posible entrenar modelos concretos, por ejemplo, con `docker compose run --rm trainer --models random_forest xgboost`. Los errores se consultan con `docker compose logs -f`. El endpoint `/ready` devuelve 503 hasta que haya un modelo registrado y accesible.

PostgreSQL y los artefactos de MLflow usan volúmenes persistentes. `docker compose down` detiene los servicios sin borrar esos datos. Las credenciales de `.env.example`, los puertos publicados y la falta de autenticación son solo para uso local. Todo despliegue requiere controles de acceso, gestión de secretos y TLS. El archivo `.env`, los datos y los artefactos no deben subirse a Git.

W&B está desactivado por defecto; MLflow funciona sin él. La integración se habilita configurando `WANDB_API_KEY` en `.env` y ejecutando:

```bash
docker compose run --rm trainer --wandb --wandb-project TU_PROYECTO
```

La integración requiere conexión y autorización de W&B. Si está activada y ocurre un error, el entrenamiento se interrumpe explícitamente. La clave no debe incluirse en comandos, logs ni código.

## Ejecución local y pruebas

La ejecución local requiere Python 3.11 o superior y una instalación de PyTorch adecuada para la plataforma:

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

El registro de modelos requiere un backend compatible con MLflow, como el PostgreSQL de Compose. SQLite también puede utilizarse para experimentos locales. La ejecución de la API fuera de Compose requiere definir `DATABASE_URL` y `MLFLOW_TRACKING_URI`; el servicio puede iniciarse con:

```bash
uvicorn rul.api:app --port 8000
```

El dashboard se inicia con `streamlit run dashboard.py`. `MODEL_NAME`, `MODEL_ALIAS` y `RUL_ALERT_THRESHOLD` deben coincidir entre el entrenamiento y el servicio. Las pruebas usan datos sintéticos y no requieren W&B ni GPU.

## API y dashboard

- `GET /health`: comprueba que el proceso está activo.
- `GET /ready`: comprueba que la base de datos y el modelo están disponibles.
- `GET /model`: devuelve el nombre, alias, versión y umbral del modelo.
- `POST /predict`: predice y guarda la ventana y el resultado en una transacción. Cada lectura requiere `cycle`, una lista `settings` de 3 valores y una lista `sensors` de 21 valores. Los ciclos deben ser consecutivos y ascendentes. Se acepta una sola lectura; en ese caso se completa la ventana repitiéndola. Por defecto, el modelo usa como máximo las últimas 30 lecturas; la longitud real se guarda con el modelo.
- `GET /units` y `GET /history/{unit}`: consultan las unidades y su historial. Cada predicción guarda la versión y el umbral utilizados.

Ejemplo mínimo, con la API y el modelo en marcha:

```bash
python - <<'PY' | curl -sS -X POST http://localhost:8000/predict -H 'Content-Type: application/json' --data-binary @-
import json
print(json.dumps({"unit": "motor-1", "readings": [
    {"cycle": 1, "settings": [0.0] * 3, "sensors": [0.0] * 21}
]}))
PY
```

Los valores son solo ilustrativos; no representan una lectura industrial real. El dashboard acepta CSV con encabezados en este orden: `unit,cycle,setting_1,setting_2,setting_3,sensor_1,...,sensor_21`. Para generar un archivo a partir de un conjunto oficial descargado:

```bash
python -c "from pathlib import Path; from rul.data import load_cycles; load_cycles(Path('data/CMAPSSData/test_FD001.txt')).to_csv('example.csv', index=False)"
```

El dashboard envía las lecturas a la API, permite elegir un motor y muestra las predicciones guardadas, la evolución de los sensores y las alertas. Si no hay modelo o base de datos disponibles, no muestra resultados simulados.
