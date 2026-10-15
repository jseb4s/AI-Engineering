# Pipeline de Extracción de Entidades Técnicas

Recibe un párrafo de texto sin procesar (un log de error o una descripción de arquitectura) y devuelve un objeto **validado con Pydantic** con las tecnologías mencionadas, el nivel de criticidad y un resumen técnico. Está construido con LangChain (LCEL), salida estructurada y reintentos automáticos.

## Ejemplo de salida

Entrada (log de error):

```text
2026-10-14 03:12:44 ERROR [api-gateway] FastAPI worker timeout after 30s.
psycopg2.OperationalError: FATAL: remaining connection slots are reserved...
Redis cache hit ratio dropped to 12%. Checkout endpoint returning HTTP 503 to all users.
```

Salida:

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL; cuello de botella en conexiones concurrentes."
}
```

## Estructura

```text
extractor-entidades/
├── schemas.py                 # Contrato Pydantic (ExtraccionTecnica)
├── chain.py                   # Modelo, prompt, cadena LCEL, reintentos, process_text()
├── main.py                    # Mini-script de prueba asíncrono (LLM real)
├── tests/test_resiliencia.py  # Pruebas sin conexión de los reintentos
├── requirements.txt
└── .env.example
```

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # y pon tu API key
```

En `.env` eliges el proveedor:

| Variable | Valores | Por defecto |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` o `openai` | `anthropic` |
| `LLM_MODEL` | nombre del modelo | `claude-haiku-4-5` / `gpt-4o-mini` |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | tu clave | — |
| `LLM_MAX_TOKENS` | tokens máximos de respuesta | `1024` |
| `MAX_INTENTOS` | intentos totales antes de fallar | `3` |
| `LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING` | `INFO` |

## Uso

Mini-script de prueba (procesa 3 textos en paralelo con `asyncio.gather`, incluido uno ambiguo como prueba de estrés):

```bash
python main.py
```

Desde tu propio código:

```python
import asyncio
from chain import process_text

resultado = asyncio.run(process_text("Kafka no consume mensajes desde el deploy..."))
print(resultado.model_dump_json(indent=2))
```

Pruebas sin conexión (no gastan tokens ni necesitan API key):

```bash
pytest -v
```

## Cómo funciona

```text
{texto} → ChatPromptTemplate → LLM.with_structured_output(ExtraccionTecnica, include_raw=True)
        → validar_salida() → ExtraccionTecnica
        └────────────── .with_retry(stop_after_attempt=3) ──────────────┘
```

### 1. Contrato (`schemas.py`)

- `tecnologias`: lista de strings con al menos un elemento. El validador quita espacios, vacíos y duplicados sin distinguir mayúsculas.
- `nivel_de_criticidad`: enum `baja | media | alta`; cualquier otro valor es rechazado.
- `resumen_tecnico`: entre 20 y 400 caracteres, con espacios normalizados.
- `extra="forbid"`: rechaza campos que el modelo invente.

### 2. Prompt template (`chain.py`)

Se usa `ChatPromptTemplate`, sin f-strings. Tiene dos variables: `{texto}` (la entrada) y `{format_instructions}`, que se inyecta con `.partial()`. Así las instrucciones de formato se pueden cambiar sin tocar la cadena.

### 3. Cadena LCEL

```python
cadena = prompt | modelo.with_structured_output(ExtraccionTecnica, include_raw=True) | RunnableLambda(validar_salida)
```

Con `include_raw=True` el modelo no lanza excepciones al fallar el parseo. En su lugar devuelve `raw`, `parsed` y `parsing_error`, lo que permite revisar el `finish_reason` antes de transformar el objeto.

### 4. Resiliencia

`validar_salida()` lanza una excepción reintentable en tres casos:

| Caso | Cómo se detecta | Excepción |
|---|---|---|
| Respuesta cortada por falta de tokens | `finish_reason == "length"` (OpenAI) o `stop_reason == "max_tokens"` (Anthropic) | `RespuestaIncompletaError` |
| JSON mal formado o que no cumple el esquema | `parsing_error` no es nulo | `OutputParserException` |
| El modelo no devolvió salida estructurada | `parsed` es nulo | `OutputParserException` |

El truncamiento se revisa **antes** del parseo: un JSON cortado a veces se puede parsear parcialmente, y no queremos aceptarlo.

`.with_retry()` envuelve la cadena completa, así que cada reintento vuelve a llamar al LLM. Usa espera exponencial con jitter y se detiene tras `MAX_INTENTOS`; si se agotan, `process_text()` registra el error y relanza la excepción.

## Logs

Ejemplo real de las pruebas, donde el modelo simulado devuelve primero un JSON roto, luego una respuesta truncada y por último una válida:

```text
INFO     pipeline | Procesando texto (14 caracteres)
WARNING  pipeline | JSON inválido o incompleto: Expecting ',' delimiter: line 1 column 42. Se reintentará si quedan intentos.
WARNING  pipeline | Respuesta truncada (finish_reason=length). Se reintentará si quedan intentos.
INFO     pipeline | Validación OK: 3 tecnologías, criticidad=alta
INFO     pipeline | Pipeline completado
```

Con `LOG_LEVEL=DEBUG` se registra también el `finish_reason` de cada llamada.

## Prueba de estrés: texto ambiguo

`main.py` incluye *"Ayer algo andaba lento con el servidor, pero luego como que se arregló solo."*, que no nombra ninguna tecnología concreta. Hay dos resultados posibles:

- El modelo extrae algo genérico (p. ej. `"servidor"`) con criticidad `baja` y el esquema lo acepta.
- El modelo devuelve una lista vacía, el validador la rechaza (`min_length=1`), se reintenta y, si sigue vacía, `process_text()` lanza la excepción tras los 3 intentos. `main.py` la captura y la muestra sin detener los demás ejemplos.

Cualquiera de los dos es un comportamiento esperado: el pipeline nunca devuelve un objeto que no cumpla el contrato.
