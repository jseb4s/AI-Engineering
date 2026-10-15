"""Contrato de salida del pipeline: el objeto que el LLM debe devolver."""
from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NivelCriticidad(str, Enum):
    """Qué tan urgente es atender lo descrito en el texto."""

    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class ExtraccionTecnica(BaseModel):
    """Entidades técnicas extraídas de un texto (log, arquitectura, incidente)."""

    model_config = ConfigDict(extra="forbid")  # rechaza campos inventados

    tecnologias: List[str] = Field(
        ...,
        min_length=1,
        description=(
            "Tecnologías, lenguajes, frameworks, bases de datos o servicios "
            "mencionados explícitamente en el texto, con su nombre oficial."
        ),
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        ...,
        description=(
            "alta: caída, pérdida de datos, seguridad o impacto en producción; "
            "media: degradación o error recuperable; "
            "baja: informativo, mejora o sin impacto."
        ),
    )
    resumen_tecnico: str = Field(
        ...,
        min_length=20,
        max_length=400,
        description="Resumen técnico de 1 a 2 frases: qué ocurre y dónde.",
    )

    @field_validator("tecnologias")
    @classmethod
    def limpiar_tecnologias(cls, valores: List[str]) -> List[str]:
        """Quita espacios, vacíos y duplicados (sin distinguir mayúsculas)."""
        vistos, limpias = set(), []
        for v in valores:
            v = v.strip()
            if v and v.lower() not in vistos:
                vistos.add(v.lower())
                limpias.append(v)
        if not limpias:
            raise ValueError("la lista de tecnologías no puede quedar vacía")
        return limpias

    @field_validator("resumen_tecnico")
    @classmethod
    def limpiar_resumen(cls, valor: str) -> str:
        return " ".join(valor.split())
