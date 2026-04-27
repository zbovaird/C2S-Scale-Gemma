"""Build a content-level review report for exported validation artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from eval.validation_artifact_review import (
    build_validation_artifact_review,
    write_validation_artifact_review,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review exported validation artifacts for scientific interpretation readiness."
    )
    parser.add_argument(
        "--artifact-manifest",
        required=True,
        help="Path to validation_artifact_manifest.json from c2s-validation-export-all.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to validation_artifact_review.json next to the manifest.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.artifact_manifest)
    output_path = (
        Path(args.output)
        if args.output
        else manifest_path.parent / "validation_artifact_review.json"
    )
    report = build_validation_artifact_review(manifest_path)
    write_validation_artifact_review(report, output_path)
    print(f"Wrote validation artifact review to {output_path}")


if __name__ == "__main__":
    main()
