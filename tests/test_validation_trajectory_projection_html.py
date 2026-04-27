from eval.validation_trajectory_projection_html import (
    render_validation_trajectory_projection_html,
)


def test_render_validation_trajectory_projection_html_includes_controls_and_canvas():
    html = render_validation_trajectory_projection_html(
        {
            "track_name": "human_fibroblast_oskm",
            "dataset_profile": "gse242423_human_fibroblast_oskm",
            "projection_method": "pca",
            "runs": [
                {
                    "label": "projective",
                    "alignment_mode": "projective_distance",
                    "rows": [
                        {
                            "cell_id": "c1",
                            "timepoint": "D2",
                            "branch_label": "productive",
                            "baseline_x": 0.0,
                            "baseline_y": 0.0,
                            "perturbed_x": 1.0,
                            "perturbed_y": 1.0,
                            "longevity_safe_zone": True,
                        }
                    ],
                }
            ],
        }
    )

    assert "<title>human_fibroblast_oskm</title>" in html
    assert "Trajectory Projection Viewer" in html
    assert 'select id="run-select"' in html
    assert 'canvas id="plot"' in html


def test_render_validation_trajectory_projection_html_escapes_title_and_legend_text():
    html = render_validation_trajectory_projection_html(
        {
            "track_name": "<unsafe>",
            "projection_method": "shared_pca",
            "runs": [],
        }
    )

    assert "<title>&lt;unsafe&gt;</title>" in html
    assert "labelText.textContent = label;" in html
