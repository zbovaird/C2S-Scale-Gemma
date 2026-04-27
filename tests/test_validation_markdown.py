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
            "trajectory_geometry_summary": [],
            "timepoint_summaries": {},
            "timepoint_comparison": [],
        },
    )

    markdown = output_path.read_text(encoding="utf-8")
    assert "## Interpretation Limits" in markdown
    assert "Representation-level evidence only." in markdown


def test_write_markdown_summary_includes_geometry_backend_metadata(tmp_path):
    output_path = tmp_path / "VALIDATION_BENCHMARK.md"
    write_markdown_summary(
        output_path,
        {
            "track_name": "human_fibroblast_oskm",
            "dataset_profile": "gse242423_human_fibroblast_oskm",
            "primary_metrics": ["safe_fraction"],
            "recommendation": {},
            "interpretation_limits": [],
            "runs": [
                {
                    "label": "projective",
                    "alignment_mode": "projective_distance",
                    "mean_l2_shift": 0.4,
                    "mean_cosine_similarity": 0.8,
                    "productive_fraction": 0.7,
                    "safe_fraction": 0.5,
                    "risk_fraction": 0.1,
                    "geometry_distance_backend": "projective_uhg_distance",
                    "geometry_fallback_used": False,
                    "alignment_graph_source": "hyperbolic_embeddings",
                }
            ],
            "trajectory_geometry_summary": [
                {
                    "label": "projective",
                    "alignment_mode": "projective_distance",
                    "mean_geometry_distance": 0.4,
                    "max_geometry_distance": 0.8,
                    "geometry_distance_backend": "projective_uhg_distance",
                    "geometry_fallback_used": False,
                }
            ],
            "timepoint_summaries": {},
            "timepoint_comparison": [],
        },
    )

    markdown = output_path.read_text(encoding="utf-8")
    assert "geometry_backend=projective_uhg_distance" in markdown
    assert "geometry_fallback_used=False" in markdown
    assert "alignment_graph_source=hyperbolic_embeddings" in markdown
    assert "## Trajectory Geometry Summary" in markdown
    assert "mean_geometry_distance=0.4000" in markdown
