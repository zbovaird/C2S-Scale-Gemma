import numpy as np

from eval.embedding_shift import build_embedding_shift_frame, summarize_embedding_shift


def test_summarize_embedding_shift_returns_expected_stats():
    baseline = np.array([[1.0, 0.0], [0.0, 1.0]])
    perturbed = np.array([[2.0, 0.0], [0.0, 2.0]])

    summary = summarize_embedding_shift(baseline, perturbed)

    assert summary["n_cells"] == 2
    assert summary["embedding_dim"] == 2
    assert summary["mean_l2_shift"] == 1.0
    assert summary["max_l2_shift"] == 1.0
    assert summary["mean_cosine_similarity"] == 1.0


def test_build_embedding_shift_frame_attaches_metadata():
    baseline = np.array([[1.0, 0.0], [0.0, 1.0]])
    perturbed = np.array([[1.0, 1.0], [1.0, 1.0]])

    rows = build_embedding_shift_frame(
        baseline,
        perturbed,
        metadata={"cell_id": ["c1", "c2"]},
    )

    assert rows[0]["cell_id"] == "c1"
    assert rows[1]["cell_id"] == "c2"
    assert rows[0]["l2_shift"] > 0
