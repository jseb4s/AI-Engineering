"""Pipeline LCEL: prompt | LLM con salida estructurada | validación, con reintentos."""
import logging
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ValidationError

from schemas import ExtraccionTecnica

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

MAX_INTENTOS = int(os.getenv("MAX_INTENTOS", "3"))


class RespuestaIncompletaError(Exception):
    """El LLM cortó la respuesta (finish_reason = length / max_tokens)."""


# ---------------------------------------------------------------------------
# 1. Cliente LLM (OpenAI o Anthropic según variables de entorno)
# ---------------------------------------------------------------------------
def crear_modelo() -> BaseChatModel:
    proveedor = os.getenv("LLM_PROVIDER", "anthropic").lower()
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))

    if proveedor == "openai":
        from langchain_openai import ChatOpenAI

        modelo = os.getenv("LLM_MODEL", "gpt-4o-mini")
        llm = ChatOpenAI(model=modelo, temperature=0, max_tokens=max_tokens, timeout=30)
    elif proveedor == "anthropic":
        from langchain_anthropic import ChatAnthropic

        modelo = os.getenv("LLM_MODEL", "claude-haiku-4-5")
        llm = ChatAnthropic(model=modelo, temperature=0, max_tokens=max_tokens, timeout=30)
    else:
        raise ValueError(f"LLM_PROVIDER no soportado: {proveedor}")

    logger.info("Modelo configurado: %s / %s (max_tokens=%s)", proveedor, modelo, max_tokens)
    return llm


# ---------------------------------------------------------------------------
# 2. Prompt template modular (sin f-strings)
# ---------------------------------------------------------------------------
SYSTEM = (
    "Eres un analista técnico senior. Extraes entidades técnicas de textos sin "
    "procesar, como descripciones de arquitectura o logs de error.\n\n"
    "Instrucciones de formato:\n{format_instructions}"
)

FORMAT_INSTRUCTIONS = (
    "- tecnologias: solo las mencionadas explícitamente; no inventes ninguna. "
    "Usa el nombre oficial (p. ej. 'PostgreSQL', no 'postgres').\n"
    "- nivel_de_criticidad: exactamente uno de 'baja', 'media' o 'alta'. Si el "
    "texto es ambiguo, elige el nivel más conservador que el texto justifique.\n"
    "- resumen_tecnico: 1 a 2 frases en español, entre 20 y 400 caracteres."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        ("human", "Texto a analizar:\n<texto>\n{texto}\n</texto>"),
    ]
).partial(format_instructions=FORMAT_INSTRUCTIONS)


# ---------------------------------------------------------------------------
# 3. Validación de la respuesta (finish_reason + errores de parseo)
# ---------------------------------------------------------------------------
def _finish_reason(raw: Any) -> Optional[str]:
    """OpenAI usa 'finish_reason'; Anthropic usa 'stop_reason'."""
    meta = getattr(raw, "response_metadata", None) or {}
    return meta.get("finish_reason") or meta.get("stop_reason")


def validar_salida(resultado: Dict[str, Any]) -> ExtraccionTecnica:
    """Recibe {'raw', 'parsed', 'parsing_error'} y devuelve el objeto validado.

    Lanza una excepción reintentable si la respuesta está cortada o mal formada.
    """
    raw = resultado.get("raw")
    razon = _finish_reason(raw)
    logger.debug("finish_reason=%s", razon)

    # Primero: si el modelo se quedó sin tokens, el JSON está incompleto.
    if razon in ("length", "max_tokens"):
        logger.warning("Respuesta truncada (finish_reason=%s). Se reintentará si quedan intentos.", razon)
        raise RespuestaIncompletaError(f"respuesta cortada por {razon}")

    error = resultado.get("parsing_error")
    if error is not None:
        logger.warning("JSON inválido o incompleto: %s. Se reintentará si quedan intentos.", str(error).splitlines()[0])
        raise OutputParserException(f"Falló el parseo/validación: {error}")

    parsed = resultado.get("parsed")
    if parsed is None:
        logger.warning("El modelo no devolvió salida estructurada. Se reintentará si quedan intentos.")
        raise OutputParserException("El modelo no devolvió salida estructurada")

    # Validación final con Pydantic (cubre proveedores que devuelven dict).
    objeto = parsed if isinstance(parsed, ExtraccionTecnica) else ExtraccionTecnica.model_validate(parsed)
    logger.info(
        "Validación OK: %d tecnologías, criticidad=%s",
        len(objeto.tecnologias),
        objeto.nivel_de_criticidad.value,
    )
    return objeto


# ---------------------------------------------------------------------------
# 4. Cadena LCEL con reintentos
# ---------------------------------------------------------------------------
ERRORES_REINTENTABLES = (
    OutputParserException,
    ValidationError,
    RespuestaIncompletaError,
)


def construir_cadena(modelo: Optional[BaseChatModel] = None) -> Runnable:
    """prompt | model.with_structured_output(Schema) | validación, con .with_retry()."""
    modelo = modelo or crear_modelo()

    # include_raw=True: no lanza al fallar el parseo y deja ver el finish_reason.
    llm_estructurado = modelo.with_structured_output(ExtraccionTecnica, include_raw=True)

    cadena = prompt | llm_estructurado | RunnableLambda(validar_salida)

    # El reintento envuelve toda la cadena: cada intento vuelve a llamar al LLM.
    return cadena.with_retry(
        retry_if_exception_type=ERRORES_REINTENTABLES,
        stop_after_attempt=MAX_INTENTOS,
        wait_exponential_jitter=True,
    )


_cadena: Optional[Runnable] = None


def _obtener_cadena() -> Runnable:
    global _cadena
    if _cadena is None:
        _cadena = construir_cadena()
    return _cadena


# ---------------------------------------------------------------------------
# 5. Punto de entrada asíncrono
# ---------------------------------------------------------------------------
async def process_text(text: str, cadena: Optional[Runnable] = None) -> ExtraccionTecnica:
    """Procesa un texto y devuelve una ExtraccionTecnica validada."""
    if not text or not text.strip():
        raise ValueError("El texto de entrada está vacío")

    cadena = cadena or _obtener_cadena()
    logger.info("Procesando texto (%d caracteres)", len(text))
    try:
        resultado = await cadena.ainvoke({"texto": text.strip()})
    except ERRORES_REINTENTABLES as exc:
        logger.error("Falló tras %d intentos: %s", MAX_INTENTOS, exc)
        raise
    logger.info("Pipeline completado")
    return resultado
