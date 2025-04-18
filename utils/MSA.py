from typing import List, Tuple, Dict
from utils.feature import *
from utils.error import *
from constant.chemical import AA2num
import os
import re
import string
import numpy as np
from utils.stoer_wagner_algorithm import stoer_wagner
import math
import gzip
from joblib import Parallel, delayed
import lmdb
import io
import pickle
from constant.datapath import MSADB_PATH

table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
num2AA = [
    'A','R','N','D','C',
    'Q','E','G','H','I',
    'L','K','M','F','P',
    'S','T','W','Y','V',
    'U',
    'X','-','0' # unknown, gap, padding
]
AA2num = {x:i for i,x in enumerate(num2AA)}

AA2num['B'] = 3
AA2num['J'] = 20
AA2num['O'] = 20
AA2num['U'] = 4
AA2num['Z'] = 6
# following AF3


class MSASEQ(ABC):
    """
    Sequence class in MSA
    """
    def __init__(self, 
                 sequence : str,
                 header : str, 
                 is_query : bool = False,
                 length : int | None = None
                 ):
        self.raw_sequence = sequence
        self._parse_sequence(sequence, length)
        self.db_name = None
        self.db_ID = None
        self.species = 'N/A'
        self.species_ID = None
        self.rep_ID = None
        
        if not is_query:
            self._parse_header(header)
            if self.db_name is None:
                self.db_name = "bfd"
        self.annotation = header[1:].strip() + f'({len(self.sequence)})'
        self.is_query = is_query

    def _parse_sequence(self, sequence: str, length: int | None) -> None:
        """
        Parse a UniRef sequence following AF3.
            - sequence : Amino acid sequence. np.ndarray(int64)
            - has_deletion : Binary feature indicating if there is a deletion to the left of each position in the MSA. (List[bool])
            - deletion_value : Raw deletion counts (the number of deletions to the left of each MSA position) are transformed to [0,1] using 2 π arctan d 3. (List[float])
        """

        if length is None:
            # query sequence
            length = len(sequence)
            self.sequence = np.array([AA2num[aa] for aa in sequence])
            self.has_deletion = np.zeros((length))
            self.deletion = np.zeros((length))
            return


        # 0 - match or gap; 1 - deletion
        lower_case = np.array([0 if c.isupper() or c=='-' else 1 for c in sequence])
        deletion = np.zeros((length))

        if np.sum(lower_case) > 0:
            # positions of deletions
            pos = np.where(lower_case==1)[0]

            # shift by occurrence
            lower_case = pos - np.arange(pos.shape[0])

            # position of deletions in cleaned sequence
            # and their length
            pos,num = np.unique(lower_case, return_counts=True)

            # append to the matrix of insetions
            deletion[pos] = num

        has_deletion = deletion > 0

        sequence = sequence.translate(table)
        self.sequence = np.array([AA2num[aa] for aa in sequence])
        self.has_deletion = has_deletion
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
        """
        # Pattern 1: UniRef-style header (with Tax=... and RepID=...)
        pattern1 = re.compile(
            r'^>(?P<db_name>UniRef\d+)_'
            r'(?P<db_ID>\S+).*?Tax=(?P<species>.*?)\s+TaxID=\S+\s+RepID=(?P<rep_ID>\S+)',
            re.IGNORECASE
        )
        
        # Pattern 2: Pipe-delimited UniProt header (with OS=...)
        pattern2 = re.compile(
            r'^>(?P<db_name>[^|]+)\|'
            r'(?P<db_ID>[^|]+)\|'
            r'(?P<rep_ID>[^|]+)\s+.*?OS=(?P<species>.*?)\s+(?=GN=|PE=|SV=)',
            re.IGNORECASE
        )
        
        result = None
        for pattern in (pattern1, pattern2):
            match = pattern.search(header)
            if match:
                result = match.groupdict()
                # For pattern3, assign default values for missing keys.
                if 'species' not in result or not result.get('species'):
                    result['species'] = "N/A"
                if 'rep_ID' not in result or not result.get('rep_ID'):
                    result['rep_ID'] = "N/A"
                break
            
        if result is not None : 
            for key, value in result.items():
                setattr(self, key, value)
        else :
            self.database = "bfd"
            self.database_ID = header[1:]
            self.species = "N/A"
            self.rep_ID = header[1:]
        
    def __len__(self):
        return len(self.sequence)
    
    def get_raw_sequence(self):
        return self.raw_sequence

    def get_sequence(self):
        return self.sequence
    
    def get_has_deletion(self):
        return self.has_deletion
    
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
            out = f"MSASEQ(\n"
            out += f"    sequence: {self.raw_sequence}\n"
            out += f"    db_name: {self.db_name}\n"
            out += f"    db_ID: {self.db_ID}\n"
            out += f"    species: {self.species}\n"
            out += f"    rep_ID: {self.rep_ID}\n"
            out += f")"
        else:
            out = f"MSASEQ(\n"
            out += f"    sequence: {self.raw_sequence}\n"
            out += f"    is_query: {self.is_query}\n"
            out += f")"
        return out
    
    def crop(self, crop_idx : np.ndarray):
        if len(crop_idx) == 1:
            self.sequence = np.array(self.sequence[crop_idx])
            self.has_deletion = np.array(self.has_deletion[crop_idx])
            self.deletion = np.array(self.deletion[crop_idx])
        else :
            self.sequence = self.sequence[crop_idx]
            self.has_deletion = self.has_deletion[crop_idx]
            self.deletion = self.deletion[crop_idx]

def read_msa_lmdb(key: str):
    env = lmdb.open(MSADB_PATH, readonly=True, lock=False)
    
    with env.begin() as txn:
        pickled_data = txn.get(key.encode())
    env.close()
    
    if pickled_data is None:
        print(f"No data found for key: {key}")
        return None

    # Unpickle the data to get back the binary blob.
    msa_data = pickle.loads(pickled_data)

    with gzip.GzipFile(fileobj=io.BytesIO(msa_data), mode='rb') as f:
        msa_data = f.read()
    
    return msa_data

class MSA(ABC):
    """
    Multiple Sequence Alignment (MSA) class
    """
    def __init__(self, 
                 sequence_hash : str | None, 
                 a3m_path : str,
                 signalp_path : str | None = None,
                 use_lmdb : bool = True
    ):
        self.sequence_hash = sequence_hash if sequence_hash is not None else os.path.basename(a3m_path).split('.')[0]
        self.a3m_path = a3m_path
        self._parse_a3m(use_lmdb)
            
    def _parse_a3m(self, use_lmdb : bool ) -> List[MSASEQ]:
        # Read file lines
        if not use_lmdb:
            if self.a3m_path.endswith('.gz'):
                lines = gzip.open(self.a3m_path, 'rt').readlines()
            else:
                lines = open(self.a3m_path, 'r').readlines()
        else :
            lines = read_msa_lmdb(self.sequence_hash).decode().split('\n')
        # remove empty lines
        lines = [line for line in lines if line.strip()]

        # Group header and sequence pairs.
        pairs = []
        header = None
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                header = line
            else:
                if header is None:
                    raise ValueError("Found sequence without a preceding header.")
                pairs.append((header, line))

        if not pairs:
            raise ValueError("No sequences found in the a3m file.")

        # Process the first sequence (query) sequentially.
        query_header, query_seq = pairs[0]
        query_header = f">{self.sequence_hash}" # replace the header with the sequence hash
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
        for idx, msaseq in enumerate(seqs):
            species = msaseq.get_species()
            if species not in species_to_idx:
                species_to_idx[species] = []
            species_to_idx[species].append(idx)
            profile_list.append(msaseq.get_sequence())
            deletion_list.append(msaseq.get_deletion())

        # Precompute profile and deletion_mean
        profile = np.array(profile_list)
        profile = np.eye(24)[profile]
        profile = np.mean(profile, axis=0)
        deletion_array = np.array(deletion_list)
        deletion_mean = 2 * np.arctan(deletion_array / 3) / np.pi
        deletion_mean = deletion_mean.mean(axis=0)

        # Set attributes.
        self.profile = profile
        self.deletion_mean = deletion_mean
        self.seqs = seqs
        self.species_to_idx = species_to_idx
        self.num_seqs = len(seqs)
        self.length = length
        self.shape = (self.num_seqs, self.length)

        return seqs

    def __len__(self):
        return self.num_seqs
        
    def __repr__(self):
        out = f"MSA(\n"
        out += f"    {self.num_seqs} sequences x {self.length} length\n"
        out += f"    sequence_hash: {self.sequence_hash}\n"
        out += f"    a3m_path: {self.a3m_path}\n"
        out += f")"
        return out
    
    def __getitem__(self, idx):
        return self.seqs[idx]
    
    def crop(self, crop_idx : np.ndarray):
        # if  crop_idx = np.array([crop_idx])
        try:
            self.profile = self.profile[crop_idx]
        except:
            breakpoint()
        self.deletion_mean = self.deletion_mean[crop_idx]
        for seq in self.seqs:
            seq.crop(crop_idx)
        new_seqs = []
        new_species_to_idx = {}

        for ii in range(self.num_seqs):
            seq = self.seqs[ii]
            raw_seq = seq.get_raw_sequence()
            if raw_seq == "-" * len(raw_seq):
                continue
            species = seq.get_species()
            if species not in new_species_to_idx:
                new_species_to_idx[species] = []
            new_species_to_idx[species].append(ii)
            new_seqs.append(seq)
        self.length = len(new_seqs[0])
        self.seqs = new_seqs
        self.species_to_idx = new_species_to_idx
        self.num_seqs = len(new_seqs)
        self.shape = (self.num_seqs, self.length)

    def get_query_sequence(self):
        return self.seqs[0].get_raw_sequence()

class ComplexMSA(ABC):
    def __init__(self, 
                 MSAs : List[MSA],
                 use_sw : bool = True
                 ):
        '''
        paired by
        1. same rep_ID
        2. species_ID
        '''
        self.num_of_MSAs = len(MSAs)
        self.use_sw = use_sw
        self._prepare_MSA(MSAs)
    
    def _test_uniqueness(self, input_dict: Dict[int, List[int]]) -> bool:
        """
        Test the uniqueness of the values in a dictionary for each key.
        """
        out = True
        for key, value in input_dict.items():
            # remove -1
            value = [v for v in value if v != -1]
            out = (len(value) == len(set(value))) and out
            if not out:
                raise ValueError("The values in the dictionary are not unique for each")

    def _pairing_MSAs(self, MSAs : Dict[int, MSA], to_removed_species : set | None = None) -> Tuple[Dict[int, List[int]], set, int]:
        species_to_idx_dict = {ii : MSA.species_to_idx for ii, MSA in MSAs.items()}

        common_species = set.intersection(*(set(d.keys()) for d in species_to_idx_dict.values()))
        common_species.discard('N/A')
        if to_removed_species is not None:
            common_species = common_species - to_removed_species

        msa_indices = {key : [] for key in MSAs.keys()}
        for species in common_species:
            min_num_seqs = min(len(species_to_idx[species]) for species_to_idx in species_to_idx_dict.values())
            for key, species_to_idx in species_to_idx_dict.items():
                msa_indices[key].extend(species_to_idx[species][:min_num_seqs])

        # sort by sum of indices (row)
        sum_of_incides = np.array([indices for indices in msa_indices.values()])
        sum_of_incides = np.sum(sum_of_incides, axis=0)
        sorted_indices = np.argsort(sum_of_incides)
        num_of_seqs = len(sorted_indices)

        msa_indices = {key : [indices[ii] for ii in sorted_indices] for key, indices in msa_indices.items()}
        self._test_uniqueness(msa_indices)

        return msa_indices, common_species, num_of_seqs
    
    def _get_chain_weight(self, MSAs : Dict[int, MSA], first_common_species: set) -> np.ndarray:
        msa_depth_list = [len(MSA) for MSA in MSAs.values()]
        species_to_idx_list = [MSA.species_to_idx for MSA in MSAs.values()]
        chain_weight = np.zeros((self.num_of_MSAs, self.num_of_MSAs)) # (i, j) : weight of MSA_i and MSA_j

        # Stoer-Wagner algorithm
        for ii in range(self.num_of_MSAs):
            hash_ii = MSAs[ii].sequence_hash
            for jj in range(ii+1, self.num_of_MSAs):
                hash_jj = MSAs[jj].sequence_hash
                if hash_ii == hash_jj:
                    chain_weight[ii, jj] = 1
                    chain_weight[jj, ii] = 1
                    continue
                species_to_idx_i = species_to_idx_list[ii]
                species_to_idx_j = species_to_idx_list[jj]

                species_i = set(species_to_idx_i.keys())
                species_j = set(species_to_idx_j.keys())
                common_species = species_i.intersection(species_j)
                common_species.discard('N/A')
                common_species = common_species - first_common_species
                if len(common_species) == 0:
                    continue

                weight_ij = 0
                for species in common_species:
                    seq_i = species_to_idx_i[species]
                    seq_j = species_to_idx_j[species]
                    weight_ij += len(seq_i) * len(seq_j) / math.sqrt(msa_depth_list[ii] * msa_depth_list[jj])
                chain_weight[ii, jj] = weight_ij
                chain_weight[jj, ii] = weight_ij

        return chain_weight

    def _prepare_MSA(self, MSAs : List[MSA]) -> None:
        msa_depth_list = [len(MSA) for MSA in MSAs]
        max_depth = max(msa_depth_list)

        MSAs = {ii : MSA for ii, MSA in enumerate(MSAs)}

        # 0. Query sequence
        query_indices = {ii : [0] for ii in MSAs.keys()}

        # 1. Simple pairing common species for all MSAs
        first_msa_indices, first_common_species, first_num_of_seqs = self._pairing_MSAs(MSAs)
        first_msa_indices = {key : query_indices[key] + indices for key, indices in first_msa_indices.items()}
        first_num_of_seqs += 1

        # 2. Stoer-Wagner algorithm MSA pairing (2 clusters)
        chain_weight = self._get_chain_weight(MSAs, first_common_species)
        set_A, set_B, cut_weight = stoer_wagner(chain_weight)
        list_A = list(set_A)
        list_B = list(set_B)
        msa_A = {ii : MSAs[ii] for ii in list_A}
        msa_B = {ii : MSAs[ii] for ii in list_B}

        sw_msa_indces_A, sw_common_speceis_A, sw_num_of_seqs_A = self._pairing_MSAs(msa_A, first_common_species)
        sw_msa_indces_B, sw_common_speceis_B, sw_num_of_seqs_B = self._pairing_MSAs(msa_B, first_common_species)

        assert sw_common_speceis_A.intersection(sw_common_speceis_B) == set()

        min_sw_pair_num = min(sw_num_of_seqs_A, sw_num_of_seqs_B)

        if min_sw_pair_num < sw_num_of_seqs_A:
            sw_msa_indces_A = {key : indices[:min_sw_pair_num] for key, indices in sw_msa_indces_A.items()}
        if min_sw_pair_num < sw_num_of_seqs_B:
            sw_msa_indces_B = {key : indices[:min_sw_pair_num] for key, indices in sw_msa_indces_B.items()}

        paired_msa_indices = {key : [] for key in MSAs.keys()}

        for key in MSAs.keys():
            if key in list_A:
                paired_msa_indices[key] = first_msa_indices[key] + sw_msa_indces_A[key]
            else:
                paired_msa_indices[key] = first_msa_indices[key] + sw_msa_indces_B[key]

        self._test_uniqueness(paired_msa_indices)

        # 3. Add extra MSAs
        final_msa_indices = {}
        for key in MSAs.keys():
            msa_depth = len(MSAs[key])
            full_indices = [ii for ii in range(msa_depth)]
            paired_indices = paired_msa_indices[key]

            # add missing indices at the end
            missing_indices = set(full_indices) - set(paired_indices)
            missing_indices = sorted(list(missing_indices))

            # if msa_depth < max_depth, add -1 to the end
            if msa_depth < max_depth:
                missing_indices += [-1] * (max_depth - msa_depth)
            final_msa_indices[key] = paired_indices + missing_indices
        self._test_uniqueness(final_msa_indices)

        final_annotation = []
        final_sequence = []
        final_has_deletion = []
        final_deletion = []

        for ii in range(max_depth):
            annotations = []
            seqs = []
            has_deletion = []
            deletion = []
            for key in MSAs.keys():
                idx = final_msa_indices[key][ii]
                if idx == -1:
                    annotations.append('N/A')
                    seqs.append(np.full((MSAs[key].length), AA2num['-'])) 
                    has_deletion.append(np.zeros((MSAs[key].length)))
                    deletion.append(np.zeros((MSAs[key].length)))
                else:
                    annotations.append(MSAs[key][idx].get_annotation())
                    seqs.append(MSAs[key][idx].get_sequence())
                    has_deletion.append(MSAs[key][idx].get_has_deletion())
                    deletion.append(MSAs[key][idx].get_deletion())
            annotations = " | ".join(annotations)
            try:
                seqs = np.concatenate(seqs)
            except:
                breakpoint()
            has_deletion = np.concatenate(has_deletion)
            deletion = np.concatenate(deletion)
            final_annotation.append(annotations)
            final_sequence.append(seqs)
            final_has_deletion.append(has_deletion)
            final_deletion.append(deletion)

        self.final_msa_indices = final_msa_indices
        self.final_annotation = np.array(final_annotation)
        self.final_sequence = np.array(final_sequence)
        self.final_has_deletion = np.array(final_has_deletion)
        self.final_deletion_value = 2 * np.arctan(np.array(final_deletion) / 3) / np.pi
        self.num_of_paired = first_num_of_seqs
        self.num_of_sw = min_sw_pair_num
        self.num_of_unpaired = max_depth - first_num_of_seqs - min_sw_pair_num
        self.total_depth = max_depth

    def to_a3m(self, annotations: List[str], sequence: np.ndarray, save_path : str):
        if annotations is None:
            annotations = self.final_annotation
        if sequence is None:
            sequence = self.final_sequence
        out = ""
        for ii in range(len(annotations)):
            out += f">{annotations[ii]}\n"
            seq = sequence[ii].tolist()
            seq = [num2AA[aa] for aa in seq]
            seq = ''.join(seq)
            out += f"{seq}\n"

        with open(save_path, 'w') as f:
            f.write(out)


    def sample(self, 
               max_msa_depth : int = 256,
               ratio : Tuple[float, float, float] = (0.5, 0.25, 0.25)
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        if self.total_depth < max_msa_depth:
            max_msa_depth = self.total_depth
        sampled = [int(ratio[ii] * max_msa_depth) for ii in range(3)]
        if sum(sampled) != max_msa_depth:
            sampled[0] += 1 # make sure the sum is equal to max_msa_depth

        if not self.use_sw:
            sampled[2] += sampled[1]
            sampled[1] = 0

        to_be_sampled = (self.num_of_paired, self.num_of_sw, self.num_of_unpaired)
        if to_be_sampled[0] < sampled[0]:
            sampled[1] += sampled[0] - to_be_sampled[0]
            sampled[0] = to_be_sampled[0]
        if to_be_sampled[1] < sampled[1]:
            sampled[2] += sampled[1] - to_be_sampled[1]
            sampled[1] = to_be_sampled[1]

        query = np.array([0])
        first_sampled = np.random.choice(self.num_of_paired, sampled[0] - 1, replace=False) # -1 for query
        sw_sampled = np.random.choice(self.num_of_sw, sampled[1], replace=False) + self.num_of_paired
        unpaired_sampled = np.random.choice(self.num_of_unpaired, sampled[2], replace=False) + self.num_of_paired + self.num_of_sw

        sampled_indices = np.concatenate([query, first_sampled, sw_sampled, unpaired_sampled])
        sampled_indices = np.sort(sampled_indices)

        sampled_annotation = self.final_annotation[sampled_indices]
        sampled_sequence = self.final_sequence[sampled_indices]
        sampled_has_deletion = self.final_has_deletion[sampled_indices]
        sampled_deletion = self.final_deletion[sampled_indices]

        return sampled_indices, sampled_annotation, sampled_sequence, sampled_has_deletion, sampled_deletion

    def __repr__(self):
        out = f"ComplexMSA(\n"
        out += f"    num_of_MSAs: {self.num_of_MSAs}\n"
        out += f"    use_sw: {self.use_sw}\n"
        out += f"    num_of_paired: {self.num_of_paired}\n"
        out += f"    num_of_sw: {self.num_of_sw}\n"
        out += f"    num_of_unpaired: {self.num_of_unpaired}\n"
        out += f"    shape: {self.total_depth} x {self.final_sequence.shape[1]}\n"
        out += f")"
        return out

def cp_singlap(signal_dir):
    pass


if __name__ == "__main__":
    # a3m1_path = "./000998.a3m"
    # a3m2_path = "./000999.a3m"
    # msa1 = MSA(None, a3m1_path)
    # msa2 = MSA(None, a3m2_path)
    # paired_msa = ComplexMSA([msa1, msa2, msa2])
    # sampled_indices, sampled_annotation, sampled_sequence, sampled_has_deletion, sampled_deletion = paired_msa.sample()

    # save_path = "./sampled.a3m"
    # paired_msa.to_a3m(sampled_annotation, sampled_sequence, save_path)

    print("MSA.py is running as a script")

        



