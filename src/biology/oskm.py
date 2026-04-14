"""Centralized OSKM gene metadata and helper utilities."""

from typing import Dict, Iterable, List, Sequence


OSKM_GENE_ALIASES: Dict[str, Sequence[str]] = {
    "POU5F1": ("POU5F1", "OCT4", "OCT-4", "OCT3/4"),
    "SOX2": ("SOX2",),
    "KLF4": ("KLF4",),
    "MYC": ("MYC", "C-MYC", "CMYC"),
}

MOUSE_OSKM_GENE_ALIASES: Dict[str, Sequence[str]] = {
    "Pou5f1": ("Pou5f1", "Oct4", "Oct-4", "Oct3/4"),
    "Sox2": ("Sox2",),
    "Klf4": ("Klf4",),
    "Myc": ("Myc", "c-Myc", "cMyc"),
}


def get_oskm_gene_aliases(species: str = "human") -> Dict[str, Sequence[str]]:
    """Return canonical OSKM alias mapping for the requested species."""
    normalized = species.lower()
    if normalized in {"human", "homo sapiens"}:
        return OSKM_GENE_ALIASES
    if normalized in {"mouse", "mus musculus"}:
        return MOUSE_OSKM_GENE_ALIASES
    raise ValueError(f"Unsupported species for OSKM aliases: {species}")


def resolve_oskm_genes(
    var_names: Iterable[str],
    species: str = "human",
) -> Dict[str, str]:
    """Resolve canonical OSKM genes to the symbols present in a dataset."""
    aliases = get_oskm_gene_aliases(species)
    upper_lookup = {str(name).upper(): str(name) for name in var_names}
    resolved: Dict[str, str] = {}

    for canonical, alias_list in aliases.items():
        for alias in alias_list:
            matched = upper_lookup.get(alias.upper())
            if matched is not None:
                resolved[canonical] = matched
                break

    return resolved


def get_present_oskm_genes(var_names: Iterable[str], species: str = "human") -> List[str]:
    """Return canonical OSKM genes that are present in ``var_names``."""
    resolved = resolve_oskm_genes(var_names, species=species)
    return [gene for gene in get_oskm_gene_aliases(species) if gene in resolved]

