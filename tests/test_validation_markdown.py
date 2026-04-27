from eval.validation_markdown import write_markdown_summary


def test_write_markdown_summary_includes_interpretation_limits(tmp_path):
    output_path = tmp_path / "VALIDATION_BENCHMARK.md"
    write_markdown_summary(
        output_path,
        {
            "track_name": "human_fibroblast_oskm",
            "dataset_profile": "gse242423_human_fibroblast_oskm",
            "primary_metrics": ["safe_fraction"],
            "recommendation": {
                "status": "mixed",
                "preferred_alignment": None,
                "reason": "Needs more validation.",
            },
            "interpretation_limits": ["Representation-level evidence only."],
            "runs": [],
            "timepoint_summaries": {},
            "timepoint_comparison": [],
        },
    )

    markdown = output_path.read_text(encoding="utf-8")
    assert "## Interpretation Limits" in markdown
    assert "Representation-level evidence only." in markdown
