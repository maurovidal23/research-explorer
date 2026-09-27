# Research Explorer

ACO multi-agent heuristic exploration of scientific citation graphs via MCP.

Partiendo de un artículo semilla, el sistema explora heurísticamente el grafo de citas
(bidireccional: referencias + citantes) mediante una **colonia de agentes LLM persistentes**
que compiten y cooperan para construir la mejor **narrativa-comprensión** de una línea de
investigación.

Usa **Optimización por Colonia de Hormigas (ACO)** como motor de exploración, **agentes LLM**
para evaluación semántica, y **proveedores académicos vía MCP** (Semantic Scholar, OpenAlex,
PubMed, arXiv) para revelar el grafo incrementalmente.

## Instalación

```bash
uv sync
# o con pip:
pip install -e ".[dev]"
```

## Configuración

Copia un perfil de configuración y ajusta tus API keys:

```bash
cp config/profiles/nan_free.toml config/local.toml
```

Configura las API keys como variables de entorno. En Windows (PowerShell):

```powershell
$env:NAN_API_KEY = "sk-..."
$env:S2_API_KEY = "..."        # opcional
$env:OPENALEX_API_KEY = "..."  # requerido para OpenAlex
```

En Windows (cmd.exe):

```cmd
set NAN_API_KEY=sk-...
set S2_API_KEY=...
set OPENALEX_API_KEY=...
```

En Linux/macOS:

```bash
export NAN_API_KEY="sk-..."
export S2_API_KEY="..."        # opcional
export OPENALEX_API_KEY="..."  # requerido para OpenAlex
```

## Uso

```bash
# Explorar desde un artículo semilla (DOI) con una query descriptiva
research-explorer explore 10.1038/nrn3241 "origin of extracellular fields in the brain"

# Corto vertical single-agent (research kernel) con un perfil acotado
research-explorer explore 10.1038/nrn3241 \
  "How do extracellular fields originate and affect neuronal computation?" \
  --pipeline research-kernel \
  --config config/profiles/kernel_quick.toml

# Smoke test manual contra proveedores y LLM reales
research-explorer explore 10.1038/nrn3241 \
  "How do extracellular fields originate and affect neuronal computation?" \
  --pipeline research-kernel \
  --config config/profiles/kernel_live.toml

# Listar los servidores MCP disponibles
research-explorer mcp list

# Arrancar un servidor MCP standalone
python -m research_explorer.mcp.semantic_scholar_server
```

### Interfaz TUI (pipeline ACO)

Añade `--tui` para seguir la exploración ACO en vivo desde la terminal. El pipeline
`research-kernel` no está soportado todavía: `--tui` con ese pipeline falla antes de
abrir proveedores o bases de datos.

```bash
research-explorer explore 10.1038/nrn3241 \
  "origin of extracellular fields in the brain" \
  --tui
```

La TUI muestra cabecera (run, tiempo, presupuesto, oleada, mejor Q), pestañas por
agente, el timeline de descubrimiento/evaluación y el detalle del agente seleccionado
(actividad actual, racional de selección, contribución del paper y línea de
investigación). Navegación solo con teclado:

| Tecla | Acción |
|---|---|
| `←` / `→` | agente anterior / siguiente |
| `↑` / `↓` | paso anterior / siguiente del timeline |
| `Enter` | detalle del paso seleccionado |
| `Home` | volver a modo en vivo |
| `e` / `f` / `p` / `n` | evaluación, frontera, paper, narrativa |
| `l` / `r` / `?` | eventos, metadatos, ayuda |
| `t` | alternar paneles en terminal estrecha |
| `q` / `Ctrl+C` | salir / cancelar |

Los eventos se publican por un contrato tipado e independiente de la UI y se
proyectan de forma determinista, de modo que la misma vista se reconstruye desde el
trace durable (`data/replay.db`). El comportamiento sin `--tui` y el pipeline
`research-kernel` no cambian.

## Servidores MCP

El proyecto incluye 4 servidores MCP standalone (usables desde Claude Desktop, etc.):

| Servidor | Proveedor | Grafo de citas |
|---|---|---|
| `semantic_scholar_server` | Semantic Scholar | referencias + citantes |
| `openalex_server` | OpenAlex | referencias + citantes |
| `pubmed_server` | PubMed E-utilities | referencias + citantes |
| `arxiv_server` | arXiv preprints | no (solo búsqueda + metadatos) |

Configuración para Claude Desktop:

```json
{
  "mcpServers": {
    "semantic-scholar": {
      "command": "uv",
      "args": ["run", "python", "-m", "research_explorer.mcp.semantic_scholar_server"]
    }
  }
}
```

## Modelo formal

Ver [`docs/model.md`](docs/model.md) para la formalización completa del modelo ACO
multi-agente bidireccional.

## Próxima arquitectura

- [`docs/specs/research-control-system.md`](docs/specs/research-control-system.md) —
  arquitectura objetivo, memoria, evaluación, optimización y replay.
- [`docs/specs/research-kernel-mvp.md`](docs/specs/research-kernel-mvp.md) — primer corte
  vertical implementable y sus criterios de aceptación.
- [`docs/specs/research-experiment.md`](docs/specs/research-experiment.md) — protocolo para
  comparar agentes, memoria y políticas bajo presupuestos equivalentes.

## Licencia

MIT
