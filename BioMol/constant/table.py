from BioMol.utils.hierarchy import PolymerType
from BioMol.constant.chemical import AA2num


class ResidueTable:
    def __init__(self, molecule_type):
        self.molecule_type = molecule_type

    def _aa_table(self, aa):
        AA_unknown = 20
        if aa not in AA2num:
            return AA_unknown
        return AA2num[aa]

    def _ligand_table(self, aa):
        return 20

    def _rna_table(self, aa):
        RNA_unknown = 25
        RNA2num = {
            "A": 21,
            "U": 22,
            "G": 23,
            "C": 24,
        }
        if aa not in RNA2num:
            return RNA_unknown
        return RNA2num[aa]

    def _dna_table(self, aa):
        DNA_unkown = 30
        DNA2num = {
            "A": 26,
            "T": 27,
            "G": 28,
            "C": 29,
        }
        if aa not in DNA2num:
            return DNA_unkown
        return DNA2num[aa]

    def __getitem__(self, aa):
        if aa == "-":
            return 31
        if self.molecule_type == PolymerType.PROTEIN:
            return self._aa_table(aa)
        elif self.molecule_type == PolymerType.RNA:
            return self._rna_table(aa)
        elif self.molecule_type == PolymerType.DNA:
            return self._dna_table(aa)
        else:
            return self._ligand_table(aa)
