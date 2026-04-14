import numpy as np

from eval.reprogramming_visuals import (
    build_dataset_readiness_summary,
    build_embedding_frame,
    build_sentence_debug_frame,
    compute_oskm_score_matrix,
)


def test_build_dataset_readiness_summary_tracks_oskm_presence():
    obs = {"timepoint": ["d0", "d2", "d2"], "batch": ["a", "a", "b"]}
    summary = build_dataset_readiness_summary(
        obs,
        ["POU5F1", "SOX2", "ACTB"],
        timepoint_column="timepoint",
        batch_column="batch",
    )

    assert summary["n_cells"] == 3
    assert summary["n_timepoints"] == 2
    assert summary["oskm_present_POU5F1"] is True
    assert summary["oskm_present_MYC"] is False


def test_build_sentence_debug_frame_tracks_gene_positions():
    rows = build_sentence_debug_frame(
        ["ACTB SOX2 KLF4", "POU5F1 GAPDH MYC"],
    )

    assert rows[0]["sox2_position"] == 1
    assert rows[1]["pou5f1_position"] == 0
    assert rows[0]["myc_position"] is None


def test_build_embedding_frame_validates_metadata_lengths():
    embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
    frame = build_embedding_frame(embeddings, metadata={"label": ["a", "b"]})

    assert list(frame.keys()) == ["x", "y", "label"]
    assert frame["label"] == ["a", "b"]


def test_compute_oskm_score_matrix_sums_available_genes():
    expression = {"POU5F1": [1.0, 0.0], "SOX2": [2.0, 1.0], "ACTB": [3.0, 3.0]}
    scores = compute_oskm_score_matrix(expression)
    assert scores.tolist() == [3.0, 1.0]
