"""Command-line entrypoint for CompIntel Research."""

from __future__ import annotations

import argparse
import asyncio

from .bundle import generate_delivery_bundle
from .execution import CompIntelExecution


async def _run(query: str) -> dict:
    execution = CompIntelExecution()
    outcome = await execution.run_intent(query)
    bundle_paths = generate_delivery_bundle(outcome)
    outcome.update(bundle_paths)
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CompIntel Research analysis.")
    parser.add_argument("query", help="Research query to analyze")
    args = parser.parse_args()

    print("Running CompIntel analysis...")
    outcome = asyncio.run(_run(args.query))
    print(f"Bundle: {outcome['bundle_path']}")
    print(f"Status: {outcome['tracker']['status']}")


if __name__ == "__main__":
    main()
