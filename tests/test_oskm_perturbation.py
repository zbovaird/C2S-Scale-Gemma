import numpy as np

from biology.perturbation import (
    apply_oskm_perturbation,
    compute_cellwise_delta_summary,
    summarize_perturbation_shift,
)
from eval.reprogramming_visuals import (
    build_perturbation_comparison_frame,
    summarize_radial_shift,
)


def test_apply_oskm_perturbation_silences_resolved_genes():
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    perturbed, metadata = apply_oskm_perturbation(
        matrix,
        ["OCT4", "SOX2", "ACTB"],
        mode="silence",
    )

    assert metadata["perturbed_symbols"] == ["OCT4", "SOX2"]
    assert np.allclose(perturbed[:, :2], 0.0)
    assert np.allclose(perturbed[:, 2], matrix[:, 2])


def test_apply_oskm_perturbation_overexpresses_selected_targets():
    matrix = np.array([[1.0, 2.0, 3.0]])
    perturbed, metadata = apply_oskm_perturbation(
        matrix,
        ["POU5F1", "SOX2", "ACTB"],
        mode="overexpress",
        factor=3.0,
        target_genes=["SOX2"],
    )

    assert metadata["perturbed_symbols"] == ["SOX2"]
    assert np.allclose(perturbed[0], [1.0, 6.0, 3.0])


def test_summarize_perturbation_shift_reports_gene_deltas():
    baseline = np.array([[1.0, 2.0], [3.0, 4.0]])
    perturbed = np.array([[0.0, 2.0], [0.0, 8.0]])
    summary = summarize_perturbation_shift(
        baseline,
        perturbed,
        ["POU5F1", "SOX2"],
        ["POU5F1", "SOX2"],
    )

    assert summary[0]["gene"] == "POU5F1"
    assert summary[0]["delta"] == -2.0
    assert summary[1]["delta"] == 2.0


def test_compute_cellwise_delta_summary_tracks_basic_statistics():
    summary = compute_cellwise_delta_summary([1.0, 2.0], [2.0, 5.0])
    assert summary["mean_delta"] == 2.0
    assert summary["max_delta"] == 3.0
    assert summary["min_delta"] == 1.0


def test_build_perturbation_comparison_frame_and_radial_summary():
    rows = build_perturbation_comparison_frame(
        [1.0, 2.0],
        [1.5, 1.0],
        metadata={"cell_type": ["fibroblast", "intermediate"]},
    )
    radial_summary = summarize_radial_shift([0.1, 0.2], [0.2, 0.1])

    assert rows[0]["delta_score"] == 0.5
    assert rows[1]["cell_type"] == "intermediate"
    assert radial_summary["n_cells"] == 2
