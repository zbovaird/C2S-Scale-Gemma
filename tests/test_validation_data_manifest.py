import json

from eval.validation_data_manifest import (
    build_candidate_dataset_rows,
    build_validation_data_manifest,
    build_download_plan,
    infer_geo_supplement_url,
    write_validation_data_manifest,
)


def test_build_validation_data_manifest_links_tracks_profiles_and_readiness():
    manifest = build_validation_data_manifest(
        track_registry={
            "validation_tracks": {
                "human_fibroblast_oskm": {
                    "title": "Human fibroblast OSKM",
                    "dataset_profile": "gse242423_human_fibroblast_oskm",
                    "accession": "GSE242423",
                    "species": "human",
                    "source_url": "https://example.test/GSE242423",
                    "baseline_data_hint": "missing_baseline.h5ad",
                    "perturbed_data_hint": "missing_perturbed.h5ad",
                    "cell_type_column": "cell_type",
                    "timepoint_column": "timepoint",
                    "expected_timepoints": ["D0", "D2"],
                    "primary_metrics": ["safe_fraction"],
                }
            }
        },
        profile_registry={
            "reprogramming_profiles": {
                "gse242423_human_fibroblast_oskm": {
                    "title": "Profile title",
                    "accession": "GSE242423",
                    "species": "human",
                    "modality": "scRNA-seq",
                    "source_url": "https://example.test/profile",
                    "baseline_data_hint": "missing_baseline.h5ad",
                    "cell_type_column": "cell_type",
                    "timepoint_column": "timepoint",
                    "notes": "Profile notes.",
                }
            }
        },
    )

    row = manifest["datasets"][0]
    assert manifest["artifact_type"] == "validation_data_manifest"
    assert row["track_name"] == "human_fibroblast_oskm"
    assert row["readiness_status"] == "needs_data"
    assert row["required_obs_columns"] == ["cell_type", "timepoint"]
    assert row["expected_timepoints"] == ["D0", "D2"]
    assert row["download_plan"]["local_path"] == "missing_baseline.h5ad"


def test_build_candidate_dataset_rows_marks_review_only_candidates():
    rows = build_candidate_dataset_rows(
        {
            "validation_dataset_candidates": {
                "waddington_ot_mef_to_ipsc": {
                    "title": "Waddington-OT MEF to iPSC",
                    "accessions": ["GSE106664"],
                    "portal_accessions": ["SCP30"],
                    "modalities": ["scRNA-seq 10x"],
                    "validation_role": "Trajectory validation baseline.",
                }
            }
        }
    )

    assert rows == [
        {
            "candidate_name": "waddington_ot_mef_to_ipsc",
            "title": "Waddington-OT MEF to iPSC",
            "study": None,
            "accessions": ["GSE106664"],
            "portal_accessions": ["SCP30"],
            "publication": None,
            "species": None,
            "trajectory": None,
            "induction": None,
            "timepoints": [],
            "modalities": ["scRNA-seq 10x"],
            "format_hints": [],
            "source_urls": [],
            "local_data_hint": None,
            "validation_role": "Trajectory validation baseline.",
            "strengths": None,
            "limitations": None,
            "notes": None,
            "acquisition_status": "candidate_review_only",
        }
    ]


def test_build_validation_data_manifest_includes_candidate_registry():
    manifest = build_validation_data_manifest(
        track_registry={
            "validation_tracks": {
                "human_fibroblast_oskm": {
                    "title": "Human fibroblast OSKM reprogramming validation",
                    "dataset_profile": "gse242423_human_fibroblast_oskm",
                    "baseline_data_hint": "data/raw/GSE242423.h5ad",
                    "perturbed_data_hint": "artifacts/GSE242423_perturbed.h5ad",
                    "cell_type_column": "cell_type",
                    "timepoint_column": "timepoint",
                    "expected_timepoints": ["D0", "iPSC"],
                    "primary_metrics": ["safe_fraction"],
                }
            },
            "validation_dataset_candidates": {
                "tabula_muris_senis": {
                    "title": "Tabula Muris Senis",
                    "accessions": ["GSE132042"],
                    "validation_role": "Aging benchmark.",
                }
            },
        },
        profile_registry={
            "reprogramming_profiles": {
                "gse242423_human_fibroblast_oskm": {
                    "accession": "GSE242423",
                    "species": "human",
                    "modality": "scRNA-seq",
                    "source_url": "https://example.test/GSE242423",
                    "baseline_data_hint": "data/raw/GSE242423.h5ad",
                    "cell_type_column": "cell_type",
                    "timepoint_column": "timepoint",
                }
            }
        },
    )

    assert manifest["datasets"][0]["track_name"] == "human_fibroblast_oskm"
    assert manifest["candidate_datasets"][0]["candidate_name"] == "tabula_muris_senis"
    assert manifest["candidate_datasets"][0]["acquisition_status"] == (
        "candidate_review_only"
    )


def test_infer_geo_supplement_url_builds_expected_bucket_url():
    url = infer_geo_supplement_url("GSE176206", "GSE176206_adipo_screen.h5ad.gz")

    assert (
        url
        == "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE176nnn/GSE176206/suppl/GSE176206_adipo_screen.h5ad.gz"
    )


def test_build_download_plan_returns_curl_command_for_geo_file():
    plan = build_download_plan(
        "GSE176206",
        "data/raw/GSE176206_adipo_screen.h5ad.gz",
    )

    assert plan["source_url"].endswith("GSE176206_adipo_screen.h5ad.gz")
    assert "curl -L" in plan["command"]
    assert "data/raw/GSE176206_adipo_screen.h5ad.gz" in plan["command"]


def test_write_validation_data_manifest_writes_json(tmp_path):
    output_path = tmp_path / "validation_data_manifest.json"
    write_validation_data_manifest({"artifact_type": "validation_data_manifest"}, output_path)

    assert json.loads(output_path.read_text())["artifact_type"] == "validation_data_manifest"
