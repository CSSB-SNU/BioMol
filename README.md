## **⚠️ Warning: This library is in **beta** version. Use at your own risk.**

## Introduction

`BioMol` is a library for loading structural data in mmCIF (from RCSB) or PDB formats (predicted structures).

- Based on Chemical Component Dictionary (CCD)
- CIF files must be obtained from the RCSB server
- Loading raw CIF can be slow (e.g., viral capsids); use precomputed LMDB for faster access
- MSA loading implemented
- Template loading not yet implemented (straightforward, but templates are not pre-fetched)
- Custom data loaders are in development (often users will implement their own loader)

## Clustering

### Structure-based Clustering

- Construct protein graphs based on contact maps
- Contact definition: min(Cα–Cα) ≤ 8.0 Å
- Two graphs are in the same split if they share at least one node

### Sequence-based Clustering

- MMseqs2 settings: sequence identity ≥ 30%, coverage ≥ 80% (`covmode 0`, `clustmode 1`)
- For antibodies (based on SAbDab):
  - If antigen present and H3 loop exists: cluster by H3 loop sequence identity ≥ 90%
  - If no H3 loop: cluster by L3 loop sequence identity ≥ 90% 

## Usage

Install via pip:

```bash
pip install git+ssh://git@github.com:CSSB-SNU/BioMol.git
```

`BioMol` is designed for machine learning training efficiency, not just data parsing. It relies on precomputed databases for speed.

Configure paths before use:

```bash
biomol configure --config_file datapath.json
```

- `datapath.json` should specify `DB_path` and `CCD_path`
- A tutorial for database construction will be provided separately

## Implemented Functions

- `biomol.choose(assembly_id, model_id, alt_id)`: select specific assembly, model, and alternate location
- `biomol.crop_and_load_msa(...)`: implements AF-Multimer cropping method and loads MSA (future versions will split this into separate functions)
- `biomol.structure.to_mmcif(path)`: save current structure to mmCIF
- `print(biomol.structure)`: display key information in a human-readable format

## Planned Features

- `biomol.help`: interactive help for BioMol
- enhanced `print(biomol)`
- `biomol.visualize`: visualize MSA and cropped regions
- `biomol.load_templates`
- loading from FASTA
- `TMscore(b1, b2)` or `b1.TMscore(b2)` for structural similarity
- `lddt(b1, b2)` or `b1.lddt(b2)` + additional comparison metrics
- `run_colabfold(b1)`, `run_ESMFold(b1)`, `run_AF3(b1)`
- all-atom clustering
- ligand tokenizer based redefinition

...and more!

## Example Code

```python
from BioMol.BioMol import BioMol

if __name__ == "__main__":
    # Automatically generate all biological assemblies
    biomol = BioMol(
        pdb_ID="9dw6",       # lowercase
        mol_types=["protein"],   # only protein
    )
    # Alternative initialization:
    # biomol = BioMol(
    #     cif="9dw6.cif",    # must be downloaded from PDB
    #     mol_types=["protein","nucleic_acid","ligand"],
    #     remove_signal_peptide=True,
    #     use_lmdb=False,      # required for NA or ligand loading
    # )

    # Select assembly, model, and alt_id
    biomol.choose("1", "1", ".")

    # Save loaded structure to mmCIF
    biomol.structure.to_mmcif("loaded_9dw6.cif")

    # Crop and load MSA
    biomol.crop_and_load_msa(
        chain_bias=('A_1',),
        interaction_bias=('A_1','C_1'),
        params={
            "method_prob": [0.0, 0.0, 1.0],  # contiguous, spatial, interface
            "crop_size": 384,
        }
    )
```

