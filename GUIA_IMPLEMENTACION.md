# Guía de implementación paso a paso — Servicio de QA extractivo por API

Esta guía lleva al equipo desde cero hasta la evaluación en vivo. El código ya está escrito en este repositorio; el trabajo de ustedes es **entrenar el modelo, levantar el servicio, probarlo, exponerlo con un túnel y completar el reporte con sus números reales**.

> **Regla de oro para la defensa (Q&A individual, 20 pts):** cada integrante debe poder explicar `qa_engine.py` y `app.py`. Lean `COMO_FUNCIONA.md`; al final trae preguntas típicas con respuesta.

---

## 0. Qué hay en el repositorio

| Archivo | Para qué sirve |
|---|---|
| `train.py` | Fine-tuning del modelo base sobre SQuAD 2.0 + cálculo del umbral de "sin respuesta" + curva de pérdida |
| `qa_engine.py` | Lógica de inferencia (ventanas, mejor span, decisión null). Lo usan el entrenamiento **y** la API |
| `app.py` | Servicio FastAPI con `/health`, `/info`, `/predict` y control de acceso por token |
| `metrics.py` | Exact Match y F1 estilo SQuAD |
| `scripts/check_service.py` | Repite las pruebas del profesor: 5 pruebas de contrato + muestra pública + latencia |
| `tests/test_offline.py` | 22 pruebas automáticas que no necesitan GPU ni modelo |
| `run_server.sh` / `run_server.ps1` | Arrancar el servidor (Mac/Linux / Windows) |
| `REPORT.md` | Reporte obligatorio (en inglés), pre-llenado; ustedes completan los `TODO` |
| `data/public_sample.json`, `docs/` | Archivos entregados por el profesor |

---

## 1. Días 1–2 · Preparar el entorno y el repositorio

**1.1 Crear el repositorio en GitHub** (uno solo para el equipo) y subir esta carpeta:

```bash
git init
git add .
git commit -m "Initial project structure: engine, API, training script"
git branch -M main
git remote add origin https://github.com/<usuario>/<repo>.git
git push -u origin main
```

> Hagan **commits pequeños y frecuentes** con mensajes claros. El historial no se puede aplastar (squash) y en el reporte deben citar el commit donde arreglaron cada problema.

**1.2 Entorno virtual en la laptop que servirá el modelo:**

```bash
# Mac / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**1.3 Verificar que todo funciona sin modelo:**

```bash
pytest -q        # deben salir 22 passed
```

**1.4 Elegir el modelo.** Recomendación: **`distilbert-base-uncased`** (66 M parámetros). Razón concreta: el modelo de referencia del profesor es DistilBERT, así que igualarlo ya da los 60 puntos completos, y es el más rápido en CPU (hay límite de 5 s por pregunta y 150 s para las 50). `roberta-base` es mejor en calidad pero ~2× más lento; solo úsenlo si tienen tiempo de sobra y miden que cabe en el tiempo.

> **Lo que hicimos:** entrenamos los dos en una RTX 4080 Laptop (35 y 62 min). RoBERTa sacó F1 83.7 vs 68.3 de DistilBERT en validación y su peor caso en CPU (~2 s) cabe en los 5 s, así que **servimos `roberta-base`**. Los logs de ambos están en `logs/<modelo>/`.

> ⚠️ Prohibido usar checkpoints ya entrenados en SQuAD (por ejemplo `distilbert-base-cased-distilled-squad`). Debe ser el modelo **base**.

---

## 2. Días 3–5 · Entrenar (fine-tuning) y calibrar el umbral

Necesitan GPU. Opciones: GPU de la universidad, una laptop con GPU NVIDIA, o Google Colab (confirmen con el profesor que lo acepta para entrenar; el servicio igual corre en su laptop).

**2.1 Instalar dependencias de entrenamiento:**

```bash
pip install -r requirements-train.txt
```

**2.2 Prueba rápida (10–20 min, confirma que no hay errores):**

```bash
python train.py --train_samples 2000 --val_samples 300 --epochs 1
```

**2.3 Entrenamiento real:**

```bash
python train.py --base_model roberta-base --epochs 2 --batch_size 16 --lr 3e-5 --logs_dir logs/roberta-base
```

Con GPU tipo T4 tarda aproximadamente 1–2 horas con el set completo (~130 k preguntas). Si no les alcanza el tiempo, usen `--train_samples 60000`.

**En Google Colab**, en una celda:

```python
!git clone https://github.com/<usuario>/<repo>.git
%cd <repo>
!pip install -q -r requirements-train.txt
!python train.py --epochs 2 --batch_size 16
!zip -r model.zip model logs
from google.colab import files; files.download("model.zip")
```

**2.4 Qué produce el entrenamiento:**

- `model/` → pesos + tokenizer + `qa_meta.json` (con el umbral elegido). **No va a git.**
- `logs/<modelo>/loss.png` → curva de pérdida (entrenamiento y validación). **Sí va a git.**
- `logs/<modelo>/train_log.json` → historial completo. **Sí va a git.**
- `logs/<modelo>/threshold.json` → EM/F1 de validación y el barrido del umbral, ponderado a 25 % sin respuesta (`--target_unanswerable`). **Sí va a git.**

**2.5 Subir los logs y compartir los pesos:**

```bash
git add logs/
git commit -m "Add training logs, loss curve and threshold results"
git push
```

Los pesos (`model.zip`) se comparten como **GitHub Release** o en una carpeta compartida (Drive). Pongan el enlace en el README.

---

## 3. Días 6–7 · Levantar la API y probarla localmente

**3.1** Copien la carpeta `model/` a la raíz del repo en la laptop servidora.

**3.2** Creen su archivo `.env` (no se sube a git):

```bash
cp .env.example .env
```

Y edítenlo: `QA_API_TOKEN=<token de desarrollo del profesor>` y `TEAM_NAME=<su nombre de equipo, sin espacios>`.

**3.3 Arrancar el servidor:**

```bash
./run_server.sh          # Mac / Linux
.\run_server.ps1         # Windows PowerShell
```

Esperen el mensaje `Model loaded ...`. Mientras carga, `/health` responde 503 (es lo correcto según el contrato).

**3.4 Probar** en otra terminal (con el mismo token cargado):

```bash
# Mac / Linux
export QA_API_TOKEN=<token>
python scripts/check_service.py --url http://localhost:8000

# Windows
$env:QA_API_TOKEN = "<token>"
python scripts/check_service.py --url http://localhost:8000
```

Deben ver `ALL CONTRACT CHECKS PASSED`, la tabla de las 5 preguntas de muestra y la latencia. Copien esa tabla al `REPORT.md` (sección 5 y 6).

**3.5 Si alguna respuesta de muestra falla**, prueben otro umbral sin reentrenar:

```bash
QA_NULL_THRESHOLD=-1.0 ./run_server.sh      # más negativo = más "null"
```

Pero no ajusten el umbral solo para las 5 de muestra: el valor de `logs/threshold.json` se eligió con 2000 preguntas y es más confiable.

---

## 4. Túnel público (necesario para el ensayo y el día de la presentación)

**4.1 Instalar ngrok** (recomendado; URL fija), crear cuenta gratis y una vez:

```bash
ngrok config add-authtoken <su authtoken de ngrok>
```

**4.2 Con el servidor corriendo**, en otra terminal:

```bash
ngrok http 8000
```

Copien la URL `https://<nombre>.ngrok-free.app`.

**4.3 Probar desde otra red** (por ejemplo, otra laptop con datos del celular):

```bash
python scripts/check_service.py --url https://<nombre>.ngrok-free.app
```

**4.4 Respaldo:** instalen también Cloudflare (`cloudflared tunnel --url http://localhost:8000`). La URL cambia cada vez; si lo usan, envíen la nueva URL al profesor.

**Checklist de la laptop servidora:** conectada a la corriente, suspensión desactivada, servidor y túnel abiertos toda la sesión, sin otros programas pesados abiertos.

---

## 5. Día 8 · Ensayo (dry run) y entrega del código

```bash
# 1. Verificar que el token NO está en el código
git grep -n "<parte del token>"      # no debe encontrar nada

# 2. Completar REPORT.md y subir
git add REPORT.md
git commit -m "Complete report"
git push

# 3. Etiqueta de entrega
git tag submission
git push origin submission
```

Envíen al profesor la URL del repositorio y la URL del túnel. Si el repo es privado, invítenlo como colaborador.

---

## 6. Días 9–10 · Presentación

1. Antes de su turno: pongan el **nuevo token** del profesor en `.env`, reinicien el servidor, verifiquen con `check_service.py`, y envíen la URL del túnel.
2. **Pitch de 3 minutos**, cubriendo los 4 criterios evaluados:
   - **Modelo:** por qué RoBERTa-base y no DistilBERT (medimos ambos: +15 F1, y aun así cabe en 5 s en CPU).
   - **Entrenamiento:** datos, hiperparámetros, umbral de null, mostrar `logs/loss.png` y EM/F1.
   - **Arquitectura del servicio:** FastAPI → token → validación → ventanas → modelo → span → JSON.
   - **Decoder:** problemas concretos que tuvieron y cómo un modelo GPT lo resolvería (y a qué costo).
3. **Q&A individual:** pueden mirar su repo en pantalla, pero sin IA. Repasen `COMO_FUNCIONA.md`.

---

## Reparto sugerido (equipo de 4–5)

| Persona | Responsabilidad |
|---|---|
| A | Entrenamiento (`train.py`), curva de pérdida, umbral |
| B | API (`app.py`), pruebas (`pytest`, `check_service.py`) |
| C | Túnel, laptop servidora, ensayo |
| D | `REPORT.md` y pitch |
| E | Investigación decoder (sección 8) y apoyo |

Todos deben leer todo el código: la pregunta del Q&A es al azar.

---

## Problemas comunes

| Síntoma | Causa probable | Solución |
|---|---|---|
| `/health` siempre 503 | El modelo no cargó | Revisar el log del servidor; ¿existe `model/` con `config.json`? |
| 401 con el token correcto | Token con espacios o comillas extra en `.env` | `QA_API_TOKEN=abc123` sin comillas ni espacios |
| `curl` falla en Windows | `curl` es alias de PowerShell | Usar `curl.exe` |
| Respuestas muy lentas | CPU ocupada o modelo grande | Cerrar programas; servir en GPU si hay; en último caso usar el modelo DistilBERT |
| `curl` a `localhost` tarda ~2 s más en Windows | `localhost` prueba IPv6 primero | Usar `127.0.0.1` |
| `ImportError: DLL load failed ... Application Control policy` al entrenar | Smart App Control de Windows bloquea DLLs de pandas 3 | `pip install "pandas<3"` (ya está en `requirements-train.txt`) |
| Casi todo da `null` | Umbral demasiado estricto | Revisar `logs/threshold.json`; probar `QA_NULL_THRESHOLD` |
| `CUDA out of memory` al entrenar | Batch muy grande | `--batch_size 8` |
