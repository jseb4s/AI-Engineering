"""Mini-script de prueba asíncrono: procesa varios textos con el LLM real.

Uso:  python main.py
"""
import asyncio
import json
import logging

from chain import process_text

logger = logging.getLogger("demo")

EJEMPLOS = {
    "log_error": (
        "2026-10-14 03:12:44 ERROR [api-gateway] FastAPI worker timeout after 30s. "
        "psycopg2.OperationalError: FATAL: remaining connection slots are reserved "
        "for non-replication superuser connections. Redis cache hit ratio dropped to 12%. "
        "Checkout endpoint returning HTTP 503 to all users."
    ),
    "arquitectura": (
        "El frontend en React consume una API GraphQL escrita en Node.js con Apollo Server. "
        "Las imágenes se guardan en Amazon S3 y los eventos se publican en Kafka para que "
        "un microservicio en Go genere reportes nocturnos. Todo corre en Kubernetes."
    ),
    # Prueba de estrés: texto ambiguo, sin tecnologías claras ni severidad explícita
    "ambiguo": "Ayer algo andaba lento con el servidor, pero luego como que se arregló solo.",
}


async def main() -> None:
    # Se procesan en paralelo para demostrar el uso asíncrono
    tareas = {nombre: process_text(texto) for nombre, texto in EJEMPLOS.items()}
    resultados = await asyncio.gather(*tareas.values(), return_exceptions=True)

    for nombre, resultado in zip(tareas, resultados):
        print(f"\n=== {nombre} ===")
        if isinstance(resultado, Exception):
            print(f"ERROR ({type(resultado).__name__}): {resultado}")
        else:
            print(json.dumps(resultado.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
