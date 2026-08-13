#!/usr/bin/env python
"""Quick-start script: explore a research line from a seed paper.

Usage:
    python scripts/explore.py <DOI> "research line description"
    python scripts/explore.py 10.1038/nrn3241 "origin of extracellular fields in the brain"
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from research_explorer.config import load_config
from research_explorer.logging_setup import configure_logging, get_logger
from research_explorer.orchestrator.runner import Orchestrator


async def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python scripts/explore.py <paper_id> <research_line_query>")
        print("Example: python scripts/explore.py 10.1038/nrn3241 'origin of extracellular fields'")
        sys.exit(1)

    seed_id = sys.argv[1]
    query = sys.argv[2]
    config_path = sys.argv[3] if len(sys.argv) > 3 else "config/default.toml"

    configure_logging("INFO")
    log = get_logger("explore")

    cfg = load_config(config_path)
    orch = Orchestrator(cfg)
    try:
        narrative = await orch.run(seed_id, query)
        print("\n" + "=" * 80)
        print("WINNING NARRATIVE")
        print("=" * 80)
        print(narrative)
    finally:
        await orch.aclose()


if __name__ == "__main__":
    asyncio.run(main())
