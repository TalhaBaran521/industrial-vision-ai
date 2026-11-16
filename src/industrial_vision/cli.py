"""Command-line interface for Industrial Vision AI."""

import argparse
import logging
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from industrial_vision.config import load_config
from industrial_vision.infrastructure.logging_config import configure_logging

LOGGER = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and report the installed application's status."""
    parser = argparse.ArgumentParser(description="Industrial Vision AI")
    parser.add_argument(
        "--version", action="version", version=version("industrial-vision-ai")
    )
    parser.add_argument("--config", type=Path, help="Path to an application TOML file.")

    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "status", help="Check the application setup and display its settings."
    )
    demo = commands.add_parser(
        "demo", help="Run a deterministic visual difference demo."
    )
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--output", type=Path, default=Path("outputs/inspections"))
    demo.add_argument("--threshold", type=int, default=25)
    demo.add_argument("--min-area", type=int, default=20)
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except (OSError, ValueError) as error:
        parser.error(f"Cannot load configuration: {error}")

    configure_logging(config.log_level)
    LOGGER.debug("Configuration loaded from %s.", args.config or "built-in defaults")
    if args.command == "demo":
        from PIL import Image

        from industrial_vision.application.inspection_service import InspectionService
        from industrial_vision.datasets.synthetic import SyntheticDatasetGenerator
        from industrial_vision.domain.entities import InspectionConfiguration
        from industrial_vision.infrastructure.result_repository import save_inspection

        try:
            settings = InspectionConfiguration(
                threshold=args.threshold, min_area=args.min_area
            )
            reference, defective, truth = SyntheticDatasetGenerator().generate(
                args.seed
            )
            artifacts = InspectionService().inspect(
                reference, defective, "synthetic.png", settings
            )
        except ValueError as error:
            parser.error(str(error))
        destination = save_inspection(artifacts, args.output)
        if destination is None:
            return 1
        try:
            Image.fromarray(defective).save(destination / "defective.png")
            Image.fromarray(truth).save(destination / "ground_truth_mask.png")
        except OSError:
            LOGGER.exception("Demo fixture save failed")
            return 1
        print(
            f"Visual difference analysis: {len(artifacts.result.defect_regions)} regions; {destination}"
        )
        return 0
    print(f"Industrial Vision AI | Phase 0 | log level: {config.log_level}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
