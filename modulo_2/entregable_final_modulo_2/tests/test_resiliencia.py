"""Pruebas sin conexión: un modelo falso simula JSON roto y respuestas truncadas."""
import os
import sys
from typing import Any, Dict, List

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chain import RespuestaIncompletaError, construir_cadena, process_text  # noqa: E402
from schemas import ExtraccionTecnica  # noqa: E402

VALIDO = {
    "tecnologias": ["FastAPI", "Redis", "PostgreSQL", "redis "],
    "nivel_de_criticidad": "alta",
    "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL; "
    "cuello de botella en conexiones concurrentes.",
}


def _respuesta(finish_reason: str = "stop", data: Any = None, error: str = None) -> Dict[str, Any]:
    parsed = None
    if data is not None and error is None:
        try:
            parsed = ExtraccionTecnica.model_validate(data)
        except ValidationError as exc:
            error = exc
    return {
        "raw": AIMessage(content="", response_metadata={"finish_reason": finish_reason}),
        "parsed": parsed,
        "parsing_error": error,
    }


class ModeloFalso:
    """Imita un chat model: cada llamada devuelve la siguiente respuesta del guion."""

    def __init__(self, guion: List[Dict[str, Any]]):
        self.guion, self.llamadas = guion, 0

    def with_structured_output(self, schema, include_raw=False):
        def responder(_):
            r = self.guion[min(self.llamadas, len(self.guion) - 1)]
            self.llamadas += 1
            return r
        return RunnableLambda(responder)


@pytest.mark.asyncio
async def test_exito_directo():
    modelo = ModeloFalso([_respuesta(data=VALIDO)])
    r = await process_text("log cualquiera", construir_cadena(modelo))
    assert r.tecnologias == ["FastAPI", "Redis", "PostgreSQL"]  # duplicado eliminado
    assert r.nivel_de_criticidad.value == "alta"
    assert modelo.llamadas == 1


@pytest.mark.asyncio
async def test_se_recupera_de_json_roto_y_truncado():
    modelo = ModeloFalso([
        _respuesta(error="Expecting ',' delimiter: line 1 column 42"),     # JSON mal formado
        _respuesta(finish_reason="length", data={"tecnologias": ["Redis"]}),  # cortado
        _respuesta(data=VALIDO),
    ])
    r = await process_text("log cualquiera", construir_cadena(modelo))
    assert r.nivel_de_criticidad.value == "alta"
    assert modelo.llamadas == 3


@pytest.mark.asyncio
async def test_se_recupera_de_validacion_pydantic():
    invalido = {**VALIDO, "nivel_de_criticidad": "urgente", "tecnologias": []}
    modelo = ModeloFalso([_respuesta(data=invalido), _respuesta(data=VALIDO)])
    r = await process_text("log", construir_cadena(modelo))
    assert r.tecnologias
    assert modelo.llamadas == 2


@pytest.mark.asyncio
async def test_falla_tras_agotar_reintentos():
    modelo = ModeloFalso([_respuesta(finish_reason="max_tokens")])
    with pytest.raises(RespuestaIncompletaError):
        await process_text("log", construir_cadena(modelo))
    assert modelo.llamadas == 3


@pytest.mark.asyncio
async def test_texto_vacio():
    with pytest.raises(ValueError):
        await process_text("   ", construir_cadena(ModeloFalso([_respuesta(data=VALIDO)])))


def test_schema_rechaza_campos_extra_y_resumen_corto():
    with pytest.raises(ValidationError):
        ExtraccionTecnica.model_validate({**VALIDO, "inventado": 1})
    with pytest.raises(ValidationError):
        ExtraccionTecnica.model_validate({**VALIDO, "resumen_tecnico": "corto"})
