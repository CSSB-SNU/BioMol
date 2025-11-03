import re
import string
from collections.abc import Callable
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from biomol.core.container import FeatureContainer
from biomol.core.feature import NodeFeature
from biomol.mapping import ResidueMapping

InputType = TypeVar("InputType", str, int, float)
FeatureType = TypeVar("FeatureType")
NumericType = TypeVar("NumericType", int, float)


def parse_sequence() -> Callable[..., dict[str, str | NDArray]]:
    """Parse a sequence string into a list of residue symbols."""

    def _worker(
        raw_sequences: list[str],
        a3m_type: str = "protein",
    ) -> dict[str, str | NDArray]:
        table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

        residue_mapping = ResidueMapping()
        max_idx = residue_mapping.MAX_INDEX

        match a3m_type.lower():
            case "protein":
                mapping_view = residue_mapping.protein
            case "rna":
                mapping_view = residue_mapping.rna
            case _:
                msg = f"Unsupported a3m_type: {a3m_type}"
                raise ValueError(msg)

        query_sequence = raw_sequences[0]
        length = len(query_sequence)

        sequences = []
        deletions = []

        for raw_sequence in raw_sequences:
            lower_case = np.array(
                [0 if c.isupper() or c == "-" else 1 for c in raw_sequence],
            )
            deletion = np.zeros(length, np.uint8)

            if np.sum(lower_case) > 0:
                # positions of deletions
                pos = np.where(lower_case == 1)[0]

                # shift by occurrence
                lower_case = pos - np.arange(pos.shape[0])

                # position of deletions in cleaned sequence
                # and their length
                pos, num = np.unique(lower_case, return_counts=True)

                # append to the matrix of insetions
                deletion[pos] = np.clip(num, 0, 255).astype(np.uint8)  # to save memory

            sequence = raw_sequence.translate(table)
            sequence = mapping_view.map(np.array(list(sequence)))
            sequences.append(sequence)
            deletions.append(deletion)
        query_sequence = np.array(list(query_sequence))
        sequences = np.stack(sequences).astype(np.int32)
        deletions = np.stack(deletions).astype(np.uint8)
        deletion_mean = 2 * np.arctan(deletions.astype(np.float32) / 3) / np.pi
        deletion_mean = deletion_mean.mean(axis=0).astype(np.float32)
        profile = np.eye(max_idx + 1, dtype=np.int32)[
            sequences
        ]  # for now, protein only
        profile = np.mean(profile, axis=0).astype(np.float32)
        sequences = sequences.transpose()
        deletions = deletions.transpose()

        query_sequence = NodeFeature(value=query_sequence)
        sequences = NodeFeature(value=sequences)
        deletions = NodeFeature(value=deletions)
        deletion_mean = NodeFeature(value=deletion_mean)
        profile = NodeFeature(value=profile)

        return {
            "query_sequence": query_sequence,
            "sequences": sequences,
            "deletions": deletions,
            "deletion_mean": deletion_mean,
            "profile": profile,
        }

    return _worker


def parse_headers() -> Callable[..., dict[str, list[str]]]:
    """Parse headers from a3m file."""

    def _worker(
        headers: list[str],
    ) -> dict[str, list[str]]:
        """
        Extract information from a FASTA header.

        The function supports three formats:

        1. UniRef-style header:
        Example:
        >UniRef100_W5NM83 G_PROTEIN_RECEP_F1_2 domain-containing protein n=1 Tax=Lepisosteus oculatus TaxID=7918 RepID=W5NM83_LEPOC

        Extracts:
            - db_name: "UniRef100"
            - db_id:   "W5NM83"
            - species: "Lepisosteus oculatus"
            - rep_id:  "W5NM83_LEPOC"

        2. Pipe-delimited UniProt header:
        Example:
        >tr|A0A060WKI3|A0A060WKI3_ONCMY Uncharacterized protein OS=Oncorhynchus mykiss GN=GSONMT00072548001 PE=3 SV=1

        Extracts:
            - db_name: "tr"
            - db_id:   "A0A060WKI3"
            - species: "Oncorhynchus mykiss"
            - rep_id:  "A0A060WKI3_ONCMY"

        3. BFD output header:
        Example:
        >SRR4029434_2280741
        >APCry4251928276_1046603.scaffolds.fasta_scaffold646995_1 # 3 # 410 # 1 # ID=646995_1;partial=11;start_type=Edge;rbs_motif=None;rbs_spacer=None;gc_cont=0.426
        # TODO extract species info from bfd db

        Extracts:
            - db_name: "bfd"
            - db_id:   "SRR4029434_2280741"
            - species: "N/A"
            - rep_id:  "SRR4029434_2280741"

        Returns
        -------
            A dictionary with keys "db_name", "db_id", "species", and "rep_id".
        """  # noqa: E501
        # Pattern 1: UniRef-style header (with Tax=... and RepID=...)
        pattern1 = re.compile(
            r"^>(?P<db_name>UniRef\d+)_"
            r"(?P<db_id>\S+).*?Tax=(?P<species>.*?)\s+TaxID=\S+\s+RepID=(?P<rep_id>\S+)",
            re.IGNORECASE,
        )

        # Pattern 2: Pipe-delimited UniProt header (with OS=...)
        pattern2 = re.compile(
            r"^>(?P<db_name>[^|]+)\|"
            r"(?P<db_id>[^|]+)\|"
            r"(?P<rep_id>[^|]+)\s+.*?OS=(?P<species>.*?)\s+(?=GN=|PE=|SV=)",
            re.IGNORECASE,
        )

        result = None
        database_list = []
        database_id = []
        species_list = []
        rep_id_list = []
        for ii, header in enumerate(headers):
            if ii == 0:
                database_list.append("query")
                database_id.append("query")
                species_list.append("query")
                rep_id_list.append("query")
                continue
            for pattern in (pattern1, pattern2):
                match = pattern.search(header)
                if match:
                    result = match.groupdict()
                    # For pattern3, assign default values for missing keys.
                    if "species" not in result or not result.get("species"):
                        result["species"] = "N/A"
                    if "rep_id" not in result or not result.get("rep_id"):
                        result["rep_id"] = "N/A"
                    break

            if result is not None:
                database = result.get("db_name", "N/A").lower()
                db_id = result.get("db_id", "N/A")
                species = result.get("species", "N/A")
                rep_id = result.get("rep_id", "N/A")
            else:
                """
                TODO: Extract species information from BFD database if possible.
                """
                # Pattern 3: BFD output header (default values)
                database = "bfd"
                db_id = header[1:].split()[0]  # Remove '>' and take first part
                species = "N/A"
                rep_id = db_id
            database_list.append(database)
            database_id.append(db_id)
            species_list.append(species)
            rep_id_list.append(rep_id)

        database_list = NodeFeature(value=np.array(database_list, dtype="S"))
        database_id = NodeFeature(value=np.array(database_id, dtype="S"))
        species_list = NodeFeature(value=np.array(species_list, dtype="S"))
        rep_id_list = NodeFeature(value=np.array(rep_id_list, dtype="S"))

        return {
            "database": database_list,
            "database_id": database_id,
            "species": species_list,
            "rep_id": rep_id_list,
        }

    return _worker


def build_container() -> Callable[..., dict[str, FeatureContainer]]:
    """Build a feature container from parsed a3m data."""

    def _worker(
        sequences: dict[str, str | NDArray],
        headers: dict[str, list[str]],
    ) -> dict[str, FeatureContainer]:
        residue_container = FeatureContainer(
            features=sequences,
        )
        chain_container = FeatureContainer(
            features=headers,
        )
        return {
            "residue_container": residue_container,
            "chain_container": chain_container,
        }

    return _worker
