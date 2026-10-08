import asyncio
import time

TOTAL_TIMEOUT = 2.0     # segundos para la orquestación completa
MAX_CONCURRENCY = 2    # llamadas simultáneas permitidas por el semáforo
TOTAL_LLAMADAS = 10     # llamadas que se intentan disparar en la fase 2

INICIO = time.perf_counter()


def log(mensaje: str) -> None:
    """Imprime el mensaje con el tiempo transcurrido desde el arranque."""
    print(f"[{time.perf_counter() - INICIO:5.2f}s] {mensaje}")


# --- Paso 2: simulación de llamadas -----------------------------------------

async def _llamada_simulada(nombre: str, latencia: float) -> str:
    log(f"{nombre}: inicia (latencia simulada {latencia}s)")
    await asyncio.sleep(latencia)  # no bloquea el event loop
    log(f"{nombre}: termina")
    return f"respuesta de {nombre}"


async def gpt_4_call() -> str:
    return await _llamada_simulada("gpt-4", 1.0)


async def claude_3_call() -> str:
    return await _llamada_simulada("claude-3", 1.5)


async def local_llama_call() -> str:
    # Latencia mayor que el timeout a propósito, para provocar el error.
    return await _llamada_simulada("llama-local", 3.0)


# --- Pasos 3 y 4: orquestación con gather + timeout -------------------------

async def orquestar() -> dict[str, str | None]:
    """Dispara los tres modelos a la vez con un límite de tiempo total.

    Devuelve la respuesta de cada modelo, o None si no terminó a tiempo.
    """
    llamadas = {
        "gpt-4": gpt_4_call,
        "claude-3": claude_3_call,
        "llama-local": local_llama_call,
    }
    # Se crean como tareas para poder consultar después cuáles terminaron.
    tareas = {
        nombre: asyncio.create_task(llamada(), name=nombre)
        for nombre, llamada in llamadas.items()
    }

    try:
        async with asyncio.timeout(TOTAL_TIMEOUT):
            await asyncio.gather(*tareas.values())
    except TimeoutError:
        log(f"ERROR: se superó el timeout total de {TOTAL_TIMEOUT}s; "
            "se cancelan las llamadas pendientes")

    # Las tareas que terminaron conservan su resultado; las demás quedaron
    # canceladas por el timeout.
    resultados: dict[str, str | None] = {}
    for nombre, tarea in tareas.items():
        if tarea.cancelled():
            resultados[nombre] = None
            log(f"{nombre}: sin respuesta (cancelada por timeout)")
        else:
            resultados[nombre] = tarea.result()
    return resultados


# --- Paso 5: control de flujo con semáforo ----------------------------------

async def _llamada_limitada(indice: int, semaforo: asyncio.Semaphore) -> str:
    nombre = f"llamada-{indice:02d}"
    log(f"{nombre}: en cola")
    async with semaforo:  # solo MAX_CONCURRENCY corrutinas pasan a la vez
        return await _llamada_simulada(nombre, 0.5)


async def control_de_flujo() -> list[str]:
    """Dispara TOTAL_LLAMADAS llamadas, con MAX_CONCURRENCY en ejecución."""
    semaforo = asyncio.Semaphore(MAX_CONCURRENCY)
    return await asyncio.gather(
        *(_llamada_limitada(i, semaforo) for i in range(1, TOTAL_LLAMADAS + 1))
    )


# --- Programa principal -----------------------------------------------------

async def main() -> None:
    log("== Fase 1: tres modelos en paralelo, timeout total de "
        f"{TOTAL_TIMEOUT}s ==")
    resultados = await orquestar()
    exitosas = [r for r in resultados.values() if r is not None]
    log(f"Fase 1 completada: {len(exitosas)} de {len(resultados)} "
        "modelos respondieron")

    log(f"== Fase 2: {TOTAL_LLAMADAS} llamadas, máximo "
        f"{MAX_CONCURRENCY} a la vez ==")
    respuestas = await control_de_flujo()
    log(f"Fase 2 completada: {len(respuestas)} respuestas recibidas")

    log("Programa finalizado sin interrupciones")


if __name__ == "__main__":
    asyncio.run(main())