import os
import re
import string
import numpy as np
import gzip
from joblib import Parallel, delayed
import lmdb
import io
import pickle
from collections import Counter

from BioMol.constant.chemical import AA2num, num2AA

# from BioMol.utils.stoer_wagner_algorithm import stoer_wagner
from BioMol import A3MDB_PATH, SEQ_TO_HASH_PATH

table = str.maketrans(dict.fromkeys(string.ascii_lowercase))


def load_seq_to_hash():
    return pickle.load(open(SEQ_TO_HASH_PATH, "rb"))


seq_to_hash = load_seq_to_hash()
hash_to_seq = {str(h).zfill(6): seq for seq, h in seq_to_hash.items()}


class MSASEQ:
    """
    Sequence class in MSA
    """

    def __init__(
        self,
        sequence: str,
        header: str,
        is_query: bool = False,
        length: int | None = None,
    ):
        self._parse_sequence(sequence, length)
        self.db_name = None
        self.db_ID = None
        self.species = "N/A"
        self.species_ID = None
        self.rep_ID = None

        if not is_query:
            self._parse_header(header)
            if self.db_name is None:
                self.db_name = "bfd"
        self.annotation = header[1:].strip() + f"({len(self.sequence)})"
        self.is_query = is_query

    def _parse_sequence(self, sequence: str, length: int | None) -> None:
        """
        Parse a UniRef sequence following AF3.
            - sequence : Amino acid sequence. np.ndarray(int64)
            - deletion_value :
                Raw deletion counts(the number of deletions to the left of each position)
                are transformed to [0,1] using 2 π arctan d 3. (List[float])
        """

        if length is None:
            # query sequence
            length = len(sequence)
            self.sequence = np.array([AA2num[aa] for aa in sequence], dtype=np.uint8)
            self.deletion = np.zeros(length, dtype=np.uint8)
            return

        # 0 - match or gap; 1 - deletion
        lower_case = np.array([0 if c.isupper() or c == "-" else 1 for c in sequence])
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

        sequence = sequence.translate(table)
        self.sequence = np.array([AA2num[aa] for aa in sequence], dtype=np.uint8)
        self.deletion = deletion

    def _parse_header(self, header: str) -> dict:
        """
        Extract information from a FASTA header.

        The function supports three formats:

        1. UniRef-style header:
        Example:
        >UniRef100_W5NM83 G_PROTEIN_RECEP_F1_2 domain-containing protein n=1 Tax=Lepisosteus oculatus TaxID=7918 RepID=W5NM83_LEPOC

        Extracts:
            - db_name: "UniRef100"
            - db_ID:   "W5NM83"
            - species: "Lepisosteus oculatus"
            - rep_ID:  "W5NM83_LEPOC"

        2. Pipe-delimited UniProt header:
        Example:
        >tr|A0A060WKI3|A0A060WKI3_ONCMY Uncharacterized protein OS=Oncorhynchus mykiss GN=GSONMT00072548001 PE=3 SV=1

        Extracts:
            - db_name: "tr"
            - db_ID:   "A0A060WKI3"
            - species: "Oncorhynchus mykiss"
            - rep_ID:  "A0A060WKI3_ONCMY"

        3. BFD output header:
        Example:
        >SRR4029434_2280741
        >APCry4251928276_1046603.scaffolds.fasta_scaffold646995_1 # 3 # 410 # 1 # ID=646995_1;partial=11;start_type=Edge;rbs_motif=None;rbs_spacer=None;gc_cont=0.426
        # TODO extract species info from bfd db

        Extracts:
            - db_name: "bfd"
            - db_ID:   "SRR4029434_2280741"
            - species: "N/A"
            - rep_ID:  "SRR4029434_2280741"

        Returns:
            A dictionary with keys "db_name", "db_ID", "species", and "rep_ID".
        """  # noqa: E501
        # Pattern 1: UniRef-style header (with Tax=... and RepID=...)
        pattern1 = re.compile(
            r"^>(?P<db_name>UniRef\d+)_"
            r"(?P<db_ID>\S+).*?Tax=(?P<species>.*?)\s+TaxID=\S+\s+RepID=(?P<rep_ID>\S+)",
            re.IGNORECASE,
        )

        # Pattern 2: Pipe-delimited UniProt header (with OS=...)
        pattern2 = re.compile(
            r"^>(?P<db_name>[^|]+)\|"
            r"(?P<db_ID>[^|]+)\|"
            r"(?P<rep_ID>[^|]+)\s+.*?OS=(?P<species>.*?)\s+(?=GN=|PE=|SV=)",
            re.IGNORECASE,
        )

        result = None
        for pattern in (pattern1, pattern2):
            match = pattern.search(header)
            if match:
                result = match.groupdict()
                # For pattern3, assign default values for missing keys.
                if "species" not in result or not result.get("species"):
                    result["species"] = "N/A"
                if "rep_ID" not in result or not result.get("rep_ID"):
                    result["rep_ID"] = "N/A"
                break

        if result is not None:
            for key, value in result.items():
                setattr(self, key, value)
        else:
            self.database = "bfd"
            self.database_ID = header[1:]
            self.species = "N/A"
            self.rep_ID = header[1:]

    def __len__(self):
        return len(self.sequence)

    def get_sequence(self):
        return self.sequence

    def get_deletion(self):
        return self.deletion

    def get_database(self):
        return self.db_name

    def get_database_ID(self):
        return self.db_ID

    def get_species(self):
        return self.species

    def get_rep_ID(self):
        return self.rep_ID

    def get_annotation(self):
        return self.annotation

    def __repr__(self):
        if not self.is_query:
            out = "MSASEQ(\n"
            out += f"    sequence: {self.get_sequence()}\n"
            out += f"    db_name: {self.db_name}\n"
            out += f"    db_ID: {self.db_ID}\n"
            out += f"    species: {self.species}\n"
            out += f"    rep_ID: {self.rep_ID}\n"
            out += ")"
        else:
            out = "MSASEQ(\n"
            out += f"    sequence: {self.get_sequence()}\n"
            out += f"    is_query: {self.is_query}\n"
            out += ")"
        return out

    def crop(self, crop_idx: np.ndarray):
        if len(crop_idx) == 1:
            self.sequence = np.array(self.sequence[crop_idx])
            self.deletion = np.array(self.deletion[crop_idx])
        else:
            self.sequence = self.sequence[crop_idx]
            self.deletion = self.deletion[crop_idx]


def read_msa_lmdb(key: str):
    env = lmdb.open(A3MDB_PATH, readonly=True, lock=False)

    with env.begin() as txn:
        pickled_data = txn.get(key.encode())
    env.close()

    if pickled_data is None:
        return None

    # Unpickle the data to get back the binary blob.
    msa_data = pickle.loads(pickled_data)

    with gzip.GzipFile(fileobj=io.BytesIO(msa_data), mode="rb") as f:
        msa_data = f.read()

    return msa_data


class MSA:
    """
    Multiple Sequence Alignment (MSA) class
    """

    def __init__(
        self,
        sequence_hash: str | None = None,
        a3m_path: str | None = None,
        use_lmdb: bool = True,
    ):
        assert sequence_hash is not None or a3m_path is not None, (
            "Either sequence_hash or a3m_path must be provided."
        )
        self.sequence_hash = (
            sequence_hash
            if sequence_hash is not None
            else os.path.basename(a3m_path).split(".")[0]
        )
        self.a3m_path = a3m_path
        self._parse_a3m(use_lmdb)

    def _parse_a3m(self, use_lmdb: bool) -> list[MSASEQ]:
        # Read file lines
        if not use_lmdb:
            if self.a3m_path.endswith(".gz"):
                lines = gzip.open(self.a3m_path, "rt").readlines()
            else:
                lines = open(self.a3m_path).readlines()
        else:
            data = read_msa_lmdb(self.sequence_hash)
            if data is None:
                # No MSA Ex) too short
                # in this case use just the sequence
                No_MSA = True
            else:
                lines = data.decode().split("\n")
                No_MSA = False
        if No_MSA:
            query_header = f">{self.sequence_hash}"
            query_seq = hash_to_seq[self.sequence_hash]
            query_msaseq = MSASEQ(query_seq, query_header, is_query=True, length=None)
            pairs = []
        else:
            # remove empty lines
            lines = [line for line in lines if line.strip()]

            # Group header and sequence pairs.
            pairs = []
            header = None
            for line in lines:
                line = line.strip()
                if line.startswith(">"):
                    header = line
                else:
                    if header is None:
                        raise ValueError("Found sequence without a preceding header.")
                    pairs.append((header, line))

            if not pairs:
                raise ValueError("No sequences found in the a3m file.")

            # Process the first sequence (query) sequentially.
            query_header, query_seq = pairs[0]
            query_header = (
                f">{self.sequence_hash}"  # replace the header with the sequence hash
            )
            query_msaseq = MSASEQ(query_seq, query_header, is_query=True, length=None)
        # Use the length of the query sequence for all other sequences.
        length = len(query_seq)

        # Helper function to process each remaining sequence.
        def process_seq(pair):
            header, seq = pair
            # For non-query sequences, pass the determined length.
            return MSASEQ(seq, header, is_query=False, length=length)

        # Process remaining sequences in parallel using all available cores.
        other_seq_list = Parallel(n_jobs=-1)(
            delayed(process_seq)(pair) for pair in pairs[1:]
        )

        # Combine query with the rest of the sequences.
        seqs = [query_msaseq] + other_seq_list

        # Build species index and precompute profile and deletion arrays.
        species_to_idx = {}
        profile_list = []
        deletion_list = []
        annotations = []
        for idx, msaseq in enumerate(seqs):
            species = msaseq.get_species()
            if species not in species_to_idx:
                species_to_idx[species] = []
            species_to_idx[species].append(idx)
            profile_list.append(msaseq.get_sequence())
            deletion_list.append(msaseq.get_deletion())
            annotations.append(msaseq.get_annotation())

        # Precompute profile and deletion_mean
        profile = np.array(profile_list).astype(np.int32)
        profile = np.eye(24, dtype=np.int32)[profile]  # for now, protein only
        profile = np.mean(profile, axis=0).astype(np.float32)
        deletion_array = np.array(deletion_list)
        deletion_mean = 2 * np.arctan(deletion_array / 3) / np.pi
        deletion_mean = deletion_mean.mean(axis=0).astype(np.float32)

        sequences = []
        deletion = []
        for seq in seqs:
            sequences.append(seq.get_sequence())
            deletion.append(seq.get_deletion())

        sequences = np.array(sequences)
        deletion = np.array(deletion)

        # Set attributes.
        self.num_seqs = len(seqs)
        self.profile = profile
        self.deletion_mean = deletion_mean
        self.sequences = sequences
        self.deletion = deletion
        self.annotations = np.array(annotations)
        self.species_to_idx = species_to_idx
        self.length = length
        self.shape = (self.num_seqs, self.length)

        return seqs

    def __len__(self):
        return self.num_seqs

    def __repr__(self):
        out = "MSA(\n"
        out += f"    {self.num_seqs} sequences x {self.length} length\n"
        out += f"    sequence_hash: {self.sequence_hash}\n"
        out += f"    a3m_path: {self.a3m_path}\n"
        out += ")"
        return out

    def __getitem__(self, idx):
        return self.sequences[idx]

    def crop(self, crop_idx: np.ndarray):
        # if  crop_idx = np.array([crop_idx])
        self.profile = self.profile[crop_idx]
        self.deletion_mean = self.deletion_mean[crop_idx]

        new_sequences = self.sequences[:, crop_idx]
        new_deletions = self.deletion[:, crop_idx]

        # gap_idx = np.where((new_sequences == AA2num["-"]).all(axis=1))[0]
        num_seqs = len(new_sequences)

        self.num_seqs = num_seqs
        self.length = len(new_sequences[0])
        self.sequences = new_sequences
        self.deletion = new_deletions
        self.shape = (self.num_seqs, self.length)

    def get_query_sequence(self):
        return self.sequences[0]

    def get_profile(self):
        return self.profile

    def get_deletion_mean(self):
        return self.deletion_mean


class ComplexMSA:  # TODO
    def __init__(
        self,
        MSAs: list[MSA],
        max_MSA_depth: int = 16384,
        max_paired_depth: int = 8192,  # including query
    ):
        """
        paired by
        1. same rep_ID
        2. species_ID
        """
        self.num_of_MSAs = len(MSAs)
        self.max_MSA_depth = max_MSA_depth
        self.max_paired_depth = max_paired_depth
        self._prepare_MSA(MSAs)

    def _test_uniqueness(self, input_dict: dict[int, list[int]]) -> bool:
        """
        Test the uniqueness of the values in a dictionary for each key.
        """
        out = True
        for value in input_dict.values():
            # remove -1
            value = [v for v in value if v != -1]
            out = (len(value) == len(set(value))) and out
            if not out:
                raise ValueError("The values in the dictionary are not unique for each")

    def _pairing_MSAs(
        self,
        MSAs: dict[int, MSA],
        max_paired_depth: int = 8191,
    ) -> tuple[dict[int, list[int]], set, int]:
        species_to_idx_dict = {ii: MSA.species_to_idx for ii, MSA in MSAs.items()}
        # gap_idx = np.where((new_sequences == AA2num["-"]).all(axis=1))[0]
        gap_idx_dict = {
            ii: np.where((MSA.sequences == AA2num["-"]).all(axis=1))[0]
            for ii, MSA in MSAs.items()
        }
        all_species = set.union(*(set(d.keys()) for d in species_to_idx_dict.values()))
        all_species.discard("N/A")

        species_to_count = Counter(
            species
            for s_to_idx in species_to_idx_dict.values()
            for species in s_to_idx.keys()
        )
        species_to_count = dict(species_to_count)
        species_to_count.pop("N/A", None)

        sorted_species = sorted(
            species_to_count.items(), key=lambda x: x[1], reverse=True
        )
        sorted_species = [species for species, count in sorted_species]

        msa_indices = {key: [] for key in MSAs.keys()}
        empty = np.array([], dtype=int)
        num_of_paired = 0
        paired_species = set()
        for species in sorted_species:
            valid_idx_dict = {
                ii: np.setdiff1d(
                    species_map.get(species, empty),
                    gap_idx_dict[ii],
                    assume_unique=True,
                )
                for ii, species_map in species_to_idx_dict.items()
            }
            num_seqs = {ii: len(idx) for ii, idx in valid_idx_dict.items()}
            # remove 0
            temp_list = [num_seqs[ii] for ii in num_seqs.keys() if num_seqs[ii] > 0]

            if len(temp_list) == 0:
                continue
            min_num_seqs = min(temp_list)
            for key, valid_idx in valid_idx_dict.items():
                if num_seqs[key] > 0:
                    msa_indices[key].extend(valid_idx[:min_num_seqs])
                    num_of_paired += min_num_seqs
                else:
                    msa_indices[key].extend([-1] * min_num_seqs)
            num_of_paired += min_num_seqs
            paired_species.add(species)
            if num_of_paired > max_paired_depth:
                break

        msa_indices = {key: np.array(indices) for key, indices in msa_indices.items()}

        self._test_uniqueness(msa_indices)

        # sort by sum of indices (row)
        sum_of_incides = np.array(list(msa_indices.values()))
        sum_of_incides = np.sum(sum_of_incides, axis=0)
        sorted_indices = np.argsort(sum_of_incides)
        num_of_seqs = len(sorted_indices)

        msa_indices = {
            # key: [indices[ii] for ii in sorted_indices]
            key: indices[sorted_indices]
            for key, indices in msa_indices.items()
        }
        self._test_uniqueness(msa_indices)

        return msa_indices, paired_species, num_of_seqs

    # def _get_chain_weight(
    #     self, MSAs: dict[int, MSA], first_common_species: set
    # ) -> np.ndarray:
    #     msa_depth_list = [len(MSA) for MSA in MSAs.values()]
    #     species_to_idx_list = [MSA.species_to_idx for MSA in MSAs.values()]
    #     chain_weight = np.zeros(
    #         (self.num_of_MSAs, self.num_of_MSAs)
    #     )  # (i, j) : weight of MSA_i and MSA_j

    #     # Stoer-Wagner algorithm
    #     for ii in range(self.num_of_MSAs):
    #         hash_ii = MSAs[ii].sequence_hash
    #         for jj in range(ii + 1, self.num_of_MSAs):
    #             hash_jj = MSAs[jj].sequence_hash
    #             if hash_ii == hash_jj:
    #                 chain_weight[ii, jj] = 1
    #                 chain_weight[jj, ii] = 1
    #                 continue
    #             species_to_idx_i = species_to_idx_list[ii]
    #             species_to_idx_j = species_to_idx_list[jj]

    #             species_i = set(species_to_idx_i.keys())
    #             species_j = set(species_to_idx_j.keys())
    #             common_species = species_i.intersection(species_j)
    #             common_species.discard("N/A")
    #             common_species = common_species - first_common_species
    #             if len(common_species) == 0:
    #                 continue

    #             weight_ij = 0
    #             for species in common_species:
    #                 seq_i = species_to_idx_i[species]
    #                 seq_j = species_to_idx_j[species]
    #                 weight_ij += (
    #                     len(seq_i)
    #                     * len(seq_j)
    #                     / math.sqrt(msa_depth_list[ii] * msa_depth_list[jj])
    #                 )
    #             chain_weight[ii, jj] = weight_ij
    #             chain_weight[jj, ii] = weight_ij

    #     return chain_weight

    def _prepare_MSA(self, MSAs: list[MSA]) -> None:
        MSAs = dict(enumerate(MSAs))
        msa_depth_list = [len(MSA) for MSA in MSAs.values()]
        max_msa_depth = max(msa_depth_list)
        max_msa_depth = min(max_msa_depth, self.max_MSA_depth)

        # 0. Query sequence
        query_indices = {ii: [0] for ii in MSAs.keys()}

        # 1. Simple pairing common species for all MSAs

        paired_msa_indices, paired_species, paired_num_of_seqs = self._pairing_MSAs(MSAs)
        paired_msa_indices = {
            key: np.concatenate([query_indices[key], indices])
            for key, indices in paired_msa_indices.items()
        }
        paired_num_of_seqs += 1
        self._test_uniqueness(paired_msa_indices)

        # 3. Add extra MSAs
        final_msa_indices = {}
        for key in MSAs.keys():
            msa_depth = len(MSAs[key])
            full_indices = list(range(msa_depth))
            paired_indices = paired_msa_indices[key]

            # add missing indices at the end
            missing_indices = set(full_indices) - set(paired_indices)
            missing_indices = sorted(missing_indices)

            # if msa_depth < max_msa_depth, add -1 to the end
            if msa_depth < max_msa_depth:
                missing_indices += [-1] * (max_msa_depth - msa_depth)
            else:
                missing_indices = missing_indices[: max_msa_depth - paired_num_of_seqs]
            missing_indices = np.array(missing_indices)
            final_msa_indices[key] = np.concatenate(
                [paired_msa_indices[key], missing_indices]
            ).astype(np.int32)
        self._test_uniqueness(final_msa_indices)

        final_annotation = []
        final_sequence = []
        final_deletion = []
        final_has_deletion = []

        for ii in range(max_msa_depth):
            annotations = []
            seqs = []
            deletion = []
            for key in MSAs.keys():
                idx = final_msa_indices[key][ii]  # TODO
                msa = MSAs[key]
                if idx == -1:
                    annotations.append("N/A")
                    seqs.append(np.full((msa.length), AA2num["-"]))
                    deletion.append(np.zeros(msa.length))
                else:
                    annotations.append(msa.annotations[idx].item())
                    seqs.append(msa.sequences[idx])
                    deletion.append(msa.deletion[idx])
            annotations = " | ".join(annotations)
            seqs = np.concatenate(seqs)
            deletion = np.concatenate(deletion)
            final_annotation.append(annotations)
            final_sequence.append(seqs)
            has_deletion = np.array(deletion > 0, dtype=np.uint8)
            final_deletion.append(deletion)
            final_has_deletion.append(has_deletion)

        # remove all gap sequences
        final_sequence = np.array(final_sequence)
        final_has_deletion = np.array(final_has_deletion)
        final_deletion = np.array(final_deletion)
        final_annotation = np.array(final_annotation)

        gap_idx = np.where((final_sequence == AA2num["-"]).all(axis=1))[0]
        final_sequence = np.delete(final_sequence, gap_idx, axis=0)
        final_deletion = np.delete(final_deletion, gap_idx, axis=0)
        final_has_deletion = np.delete(final_has_deletion, gap_idx, axis=0)
        final_annotation = np.delete(final_annotation, gap_idx, axis=0)
        final_msa_indices = {
            key: np.delete(indices, gap_idx, axis=0)
            for key, indices in final_msa_indices.items()
        }

        # concat profile, deletion_mean
        profile = np.concatenate([MSA.get_profile() for MSA in MSAs.values()], axis=0)
        deletion_mean = np.concatenate(
            [MSA.get_deletion_mean() for MSA in MSAs.values()], axis=0
        )

        self.msa_indices = final_msa_indices
        self.annotation = np.array(final_annotation)

        self.msa = final_sequence
        self.has_deletion = final_has_deletion
        self.deletion_value = 2 * np.arctan(final_deletion / 3) / np.pi
        self.profile = profile
        self.deletion_mean = deletion_mean

        self.num_of_paired = paired_num_of_seqs
        self.num_of_unpaired = self.msa.shape[0] - paired_num_of_seqs - len(gap_idx)
        self.total_depth = self.num_of_paired + self.num_of_unpaired

    def to_a3m(self, annotations: list[str], msa: np.ndarray, save_path: str):
        if annotations is None:
            annotations = self.annotation
        if msa is None:
            msa = self.msa
        out = ""
        for ii in range(len(annotations)):
            out += f">{annotations[ii]}\n"
            seq = msa[ii].tolist()
            seq = [num2AA[aa] for aa in seq]
            seq = "".join(seq)
            out += f"{seq}\n"

        with open(save_path, "w") as f:
            f.write(out)

    def sample(
        self,
        max_msa_depth: int = 256,
        ratio: tuple[float, float] = (0.5, 0.5),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.total_depth < max_msa_depth:
            max_msa_depth = self.total_depth
        sampled = [int(ratio[ii] * max_msa_depth) for ii in range(3)]
        if sum(sampled) != max_msa_depth:
            sampled[0] += 1  # make sure the sum is equal to max_msa_depth

        to_be_sampled = (self.num_of_paired, self.num_of_unpaired)
        if to_be_sampled[0] < sampled[0]:
            sampled[1] += sampled[0] - to_be_sampled[0]
            sampled[0] = to_be_sampled[0]

        query = np.array([0])
        paired_sampled = np.random.choice(
            self.num_of_paired, sampled[0] - 1, replace=False
        )  # -1 for query
        unpaired_sampled = (
            np.random.choice(self.num_of_unpaired, sampled[1], replace=False)
            + self.num_of_paired
        )

        sampled_indices = np.concatenate([query, paired_sampled, unpaired_sampled])
        sampled_indices = np.sort(sampled_indices)

        sampled_annotation = self.annotation[sampled_indices]
        sampled_sequence = self.msa[sampled_indices]
        sampled_has_deletion = self.has_deletion[sampled_indices]
        sampled_deletion_value = self.deletion_value[sampled_indices]

        return (
            sampled_indices,
            sampled_annotation,
            sampled_sequence,
            sampled_has_deletion,
            sampled_deletion_value,
            self.profile,
            self.deletion_mean,
        )

    def __repr__(self):
        out = "ComplexMSA(\n"
        out += f"    num_of_MSAs: {self.num_of_MSAs}\n"
        out += f"    num_of_paired: {self.num_of_paired}\n"
        out += f"    num_of_unpaired: {self.num_of_unpaired}\n"
        out += f"    shape: {self.total_depth} x {self.msa.shape[1]}\n"
        out += ")"
        return out


def cp_singlap(signal_dir):
    pass


if __name__ == "__main__":
    msa = MSA(sequence_hash="837082")  # No MSA due to short length
    breakpoint()
