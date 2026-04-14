from data.collate import GraphTextCollator
from data.dataset import Cell2SentenceDataset, CellSentenceDataset
from fusion.heads import FusionHead


def test_cell2sentence_dataset_alias_exists():
    assert issubclass(Cell2SentenceDataset, CellSentenceDataset)


def test_graph_text_collator_calls_paired_collation():
    collator = GraphTextCollator()
    batch = [
        {
            "input_ids": __import__("torch").tensor([1, 2]),
            "attention_mask": __import__("torch").tensor([1, 1]),
            "cell_id": "c1",
            "cell_type": "fibroblast",
            "tissue": "skin",
            "n_genes": 2,
            "total_counts": 10,
        }
    ]
    result = collator(batch)
    assert "input_ids" in result
    assert "cell_ids" in result


def test_fusion_head_wrapper_uses_expected_output_dimension():
    head = FusionHead(graph_dim=3, text_dim=5, fusion_dim=7)
    assert head.output_dim == 7
