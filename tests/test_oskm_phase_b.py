from biology.oskm import get_present_oskm_genes, resolve_oskm_genes
from data.dataset import CellSentenceDataset
from graphs.build_knn import compute_oskm_edge_multiplier


def test_resolve_oskm_genes_handles_aliases():
    resolved = resolve_oskm_genes(["OCT4", "SOX2", "KLF4", "CMYC", "ACTB"])
    assert resolved == {
        "POU5F1": "OCT4",
        "SOX2": "SOX2",
        "KLF4": "KLF4",
        "MYC": "CMYC",
    }


def test_get_present_oskm_genes_returns_canonical_order():
    present = get_present_oskm_genes(["MYC", "ACTB", "SOX2"])
    assert present == ["SOX2", "MYC"]


def test_apply_anchor_policy_prepends_present_oskm_genes():
    dataset = CellSentenceDataset.__new__(CellSentenceDataset)
    dataset.oskm_anchor_mode = "prepend_present"
    anchored = dataset._apply_anchor_policy(
        ["ACTB", "SOX2", "GAPDH", "KLF4"],
        ["SOX2", "KLF4", "MYC"],
    )
    assert anchored[:2] == ["SOX2", "KLF4"]
    assert anchored[2:] == ["ACTB", "GAPDH"]


def test_compute_oskm_edge_multiplier_only_upweights_dual_positive_edges():
    assert compute_oskm_edge_multiplier(1.0, 2.0, weight_multiplier=1.7) == 1.7
    assert compute_oskm_edge_multiplier(1.0, 0.0, weight_multiplier=1.7) == 1.0
